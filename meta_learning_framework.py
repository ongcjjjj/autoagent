# 自主进化Agent - 第3轮提升：元学习框架
# Meta Learning Framework - Few-shot学习与快速适应能力

import asyncio
import math
import random
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque
import logging
from abc import ABC, abstractmethod
import copy

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaLearningStrategy(Enum):
    """元学习策略枚举"""
    MODEL_AGNOSTIC = "model_agnostic"           # 模型无关元学习 (MAML)
    GRADIENT_BASED = "gradient_based"           # 基于梯度的元学习
    MEMORY_AUGMENTED = "memory_augmented"       # 记忆增强网络
    PROTOTYPE_BASED = "prototype_based"         # 原型网络
    RELATION_BASED = "relation_based"           # 关系网络
    OPTIMIZATION_BASED = "optimization_based"   # 基于优化的元学习
    METRIC_LEARNING = "metric_learning"         # 度量学习
    TRANSFER_LEARNING = "transfer_learning"     # 迁移学习

class TaskType(Enum):
    """任务类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE_MODELING = "sequence_modeling"
    REINFORCEMENT_LEARNING = "rl"
    GENERATION = "generation"
    REASONING = "reasoning"
    PLANNING = "planning"
    MULTIMODAL = "multimodal"

@dataclass
class Task:
    """任务数据结构"""
    task_id: str
    task_type: TaskType
    domain: str
    description: str
    support_set: List[Tuple[Any, Any]]  # (input, output) pairs
    query_set: List[Tuple[Any, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: float = 0.5
    similarity_features: List[float] = field(default_factory=list)

@dataclass
class MetaKnowledge:
    """元知识数据结构"""
    knowledge_id: str
    knowledge_type: str
    content: Any
    applicability_score: float
    usage_count: int = 0
    success_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    source_tasks: List[str] = field(default_factory=list)

@dataclass
class LearningExperience:
    """学习经验数据结构"""
    task_id: str
    strategy_used: MetaLearningStrategy
    performance: float
    adaptation_steps: int
    learning_curve: List[float]
    insights: List[str]
    timestamp: float = field(default_factory=time.time)

class TaskSimilarityMeasurer:
    """任务相似性度量器"""
    
    def __init__(self):
        self.similarity_cache = {}
        self.feature_weights = {
            'domain': 0.3,
            'task_type': 0.25,
            'complexity': 0.2,
            'data_structure': 0.15,
            'semantics': 0.1
        }
        
    def compute_similarity(self, task1: Task, task2: Task) -> float:
        """计算两个任务的相似性"""
        cache_key = f"{task1.task_id}_{task2.task_id}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        similarity_scores = {}
        
        # 1. 领域相似性
        similarity_scores['domain'] = 1.0 if task1.domain == task2.domain else 0.3
        
        # 2. 任务类型相似性
        similarity_scores['task_type'] = 1.0 if task1.task_type == task2.task_type else 0.2
        
        # 3. 复杂度相似性
        complexity_diff = abs(task1.difficulty - task2.difficulty)
        similarity_scores['complexity'] = max(0, 1.0 - complexity_diff)
        
        # 4. 数据结构相似性
        similarity_scores['data_structure'] = self._compute_data_similarity(task1, task2)
        
        # 5. 语义相似性
        similarity_scores['semantics'] = self._compute_semantic_similarity(task1, task2)
        
        # 加权总分
        total_similarity = sum(
            self.feature_weights[feature] * score 
            for feature, score in similarity_scores.items()
        )
        
        self.similarity_cache[cache_key] = total_similarity
        return total_similarity
        
    def _compute_data_similarity(self, task1: Task, task2: Task) -> float:
        """计算数据结构相似性"""
        if not task1.support_set or not task2.support_set:
            return 0.5
            
        # 简化的数据结构相似性计算
        sample1 = task1.support_set[0]
        sample2 = task2.support_set[0]
        
        # 比较输入输出的类型
        input_type_match = type(sample1[0]) == type(sample2[0])
        output_type_match = type(sample1[1]) == type(sample2[1])
        
        similarity = 0.0
        if input_type_match:
            similarity += 0.5
        if output_type_match:
            similarity += 0.5
            
        return similarity
        
    def _compute_semantic_similarity(self, task1: Task, task2: Task) -> float:
        """计算语义相似性"""
        desc1_words = set(task1.description.lower().split())
        desc2_words = set(task2.description.lower().split())
        
        if not desc1_words or not desc2_words:
            return 0.5
            
        intersection = desc1_words & desc2_words
        union = desc1_words | desc2_words
        
        return len(intersection) / len(union) if union else 0.0
        
    def find_similar_tasks(self, target_task: Task, task_pool: List[Task], 
                          top_k: int = 5) -> List[Tuple[Task, float]]:
        """找到最相似的k个任务"""
        similarities = []
        for task in task_pool:
            if task.task_id != target_task.task_id:
                sim = self.compute_similarity(target_task, task)
                similarities.append((task, sim))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class MetaLearningAlgorithm(ABC):
    """元学习算法抽象基类"""
    
    @abstractmethod
    def meta_train(self, tasks: List[Task]) -> Dict[str, Any]:
        """元训练过程"""
        pass
        
    @abstractmethod
    def fast_adapt(self, new_task: Task, adaptation_steps: int = 5) -> Dict[str, Any]:
        """快速适应新任务"""
        pass
        
    @abstractmethod
    def evaluate_adaptation(self, task: Task, adapted_model: Any) -> float:
        """评估适应效果"""
        pass

class ModelAgnosticMetaLearning(MetaLearningAlgorithm):
    """模型无关元学习 (MAML)"""
    
    def __init__(self, learning_rate: float = 0.01, meta_learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.meta_parameters = {}
        self.adaptation_history = []
        
    def meta_train(self, tasks: List[Task]) -> Dict[str, Any]:
        """MAML元训练"""
        logger.info(f"开始MAML元训练，任务数量: {len(tasks)}")
        
        meta_loss = 0.0
        meta_gradients = {}
        
        for task in tasks:
            # 任务特定的快速适应
            adapted_params = self._inner_loop_adaptation(task)
            
            # 计算元损失
            task_loss = self._compute_meta_loss(task, adapted_params)
            meta_loss += task_loss
            
            # 计算元梯度
            task_gradients = self._compute_meta_gradients(task, adapted_params)
            for param_name, grad in task_gradients.items():
                if param_name not in meta_gradients:
                    meta_gradients[param_name] = []
                meta_gradients[param_name].append(grad)
                
        # 更新元参数
        for param_name, grad_list in meta_gradients.items():
            avg_gradient = sum(grad_list) / len(grad_list)
            if param_name not in self.meta_parameters:
                self.meta_parameters[param_name] = random.random()
            self.meta_parameters[param_name] -= self.meta_learning_rate * avg_gradient
            
        training_result = {
            'meta_loss': meta_loss / len(tasks),
            'meta_parameters': copy.deepcopy(self.meta_parameters),
            'tasks_trained': len(tasks),
            'strategy': MetaLearningStrategy.MODEL_AGNOSTIC
        }
        
        logger.info(f"MAML元训练完成，元损失: {training_result['meta_loss']:.4f}")
        return training_result
        
    def _inner_loop_adaptation(self, task: Task) -> Dict[str, Any]:
        """内循环快速适应"""
        adapted_params = copy.deepcopy(self.meta_parameters)
        
        for step in range(5):  # 快速适应步数
            # 在支持集上计算梯度
            gradients = self._compute_task_gradients(task, adapted_params)
            
            # 更新参数
            for param_name, grad in gradients.items():
                adapted_params[param_name] -= self.learning_rate * grad
                
        return adapted_params
        
    def _compute_task_gradients(self, task: Task, params: Dict[str, Any]) -> Dict[str, Any]:
        """计算任务特定梯度（模拟）"""
        gradients = {}
        
        # 模拟梯度计算
        for param_name in params:
            # 简化的梯度计算
            loss = self._compute_task_loss(task, params)
            gradients[param_name] = random.uniform(-0.1, 0.1) * loss
            
        return gradients
        
    def _compute_task_loss(self, task: Task, params: Dict[str, Any]) -> float:
        """计算任务损失（模拟）"""
        # 基于任务复杂度和参数的模拟损失
        base_loss = task.difficulty
        param_penalty = sum(abs(p) for p in params.values()) * 0.01
        return base_loss + param_penalty + random.uniform(0, 0.1)
        
    def _compute_meta_loss(self, task: Task, adapted_params: Dict[str, Any]) -> float:
        """计算元损失"""
        # 在查询集上评估适应后的模型
        return self._compute_task_loss(task, adapted_params)
        
    def _compute_meta_gradients(self, task: Task, adapted_params: Dict[str, Any]) -> Dict[str, Any]:
        """计算元梯度"""
        meta_gradients = {}
        
        for param_name in adapted_params:
            # 模拟元梯度计算
            meta_gradients[param_name] = random.uniform(-0.05, 0.05)
            
        return meta_gradients
        
    def fast_adapt(self, new_task: Task, adaptation_steps: int = 5) -> Dict[str, Any]:
        """快速适应新任务"""
        logger.info(f"MAML快速适应任务: {new_task.task_id}")
        
        # 从元参数开始适应
        adapted_params = copy.deepcopy(self.meta_parameters)
        adaptation_curve = []
        
        for step in range(adaptation_steps):
            # 计算当前性能
            current_loss = self._compute_task_loss(new_task, adapted_params)
            adaptation_curve.append(current_loss)
            
            # 梯度更新
            gradients = self._compute_task_gradients(new_task, adapted_params)
            for param_name, grad in gradients.items():
                adapted_params[param_name] -= self.learning_rate * grad
                
        final_performance = self.evaluate_adaptation(new_task, adapted_params)
        
        adaptation_result = {
            'adapted_parameters': adapted_params,
            'adaptation_curve': adaptation_curve,
            'final_performance': final_performance,
            'adaptation_steps': adaptation_steps,
            'strategy': MetaLearningStrategy.MODEL_AGNOSTIC
        }
        
        self.adaptation_history.append(adaptation_result)
        logger.info(f"快速适应完成，最终性能: {final_performance:.4f}")
        
        return adaptation_result
        
    def evaluate_adaptation(self, task: Task, adapted_model: Any) -> float:
        """评估适应效果"""
        loss = self._compute_task_loss(task, adapted_model)
        # 转换为性能分数（越高越好）
        performance = max(0, 1.0 - loss)
        return performance

class PrototypeNetwork(MetaLearningAlgorithm):
    """原型网络元学习"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.prototype_embeddings = {}
        self.class_prototypes = {}
        
    def meta_train(self, tasks: List[Task]) -> Dict[str, Any]:
        """原型网络元训练"""
        logger.info(f"开始原型网络元训练，任务数量: {len(tasks)}")
        
        total_accuracy = 0.0
        
        for task in tasks:
            # 为每个类计算原型
            class_prototypes = self._compute_prototypes(task.support_set)
            
            # 在查询集上评估
            accuracy = self._evaluate_prototypes(task.query_set, class_prototypes)
            total_accuracy += accuracy
            
            # 存储原型信息
            self.class_prototypes[task.task_id] = class_prototypes
            
        avg_accuracy = total_accuracy / len(tasks)
        
        training_result = {
            'average_accuracy': avg_accuracy,
            'prototypes_learned': len(self.class_prototypes),
            'embedding_dim': self.embedding_dim,
            'strategy': MetaLearningStrategy.PROTOTYPE_BASED
        }
        
        logger.info(f"原型网络元训练完成，平均准确率: {avg_accuracy:.4f}")
        return training_result
        
    def _compute_prototypes(self, support_set: List[Tuple[Any, Any]]) -> Dict[Any, List[float]]:
        """计算类原型"""
        class_embeddings = defaultdict(list)
        
        # 将支持集按类别分组
        for input_data, label in support_set:
            embedding = self._embed_input(input_data)
            class_embeddings[label].append(embedding)
            
        # 计算每个类的原型（平均嵌入）
        prototypes = {}
        for class_label, embeddings in class_embeddings.items():
            if embeddings:
                prototype = [sum(dim_values) / len(embeddings) 
                           for dim_values in zip(*embeddings)]
                prototypes[class_label] = prototype
                
        return prototypes
        
    def _embed_input(self, input_data: Any) -> List[float]:
        """将输入数据嵌入到特征空间"""
        # 简化的嵌入函数
        if isinstance(input_data, str):
            # 文本嵌入
            hash_value = hash(input_data)
            embedding = [(hash_value >> i) & 1 for i in range(self.embedding_dim)]
        elif isinstance(input_data, (int, float)):
            # 数值嵌入
            embedding = [math.sin(input_data * i) for i in range(self.embedding_dim)]
        else:
            # 默认随机嵌入
            embedding = [random.random() for _ in range(self.embedding_dim)]
            
        return embedding
        
    def _evaluate_prototypes(self, query_set: List[Tuple[Any, Any]], 
                           prototypes: Dict[Any, List[float]]) -> float:
        """在查询集上评估原型"""
        if not query_set or not prototypes:
            return 0.0
            
        correct_predictions = 0
        
        for input_data, true_label in query_set:
            query_embedding = self._embed_input(input_data)
            
            # 找到最近的原型
            min_distance = float('inf')
            predicted_label = None
            
            for class_label, prototype in prototypes.items():
                distance = self._euclidean_distance(query_embedding, prototype)
                if distance < min_distance:
                    min_distance = distance
                    predicted_label = class_label
                    
            if predicted_label == true_label:
                correct_predictions += 1
                
        accuracy = correct_predictions / len(query_set)
        return accuracy
        
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """计算欧几里得距离"""
        if len(vec1) != len(vec2):
            return float('inf')
            
        distance = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
        return math.sqrt(distance)
        
    def fast_adapt(self, new_task: Task, adaptation_steps: int = 1) -> Dict[str, Any]:
        """原型网络快速适应"""
        logger.info(f"原型网络快速适应任务: {new_task.task_id}")
        
        # 原型网络通常只需要一步适应
        prototypes = self._compute_prototypes(new_task.support_set)
        performance = self._evaluate_prototypes(new_task.query_set, prototypes)
        
        adaptation_result = {
            'prototypes': prototypes,
            'performance': performance,
            'adaptation_steps': 1,
            'strategy': MetaLearningStrategy.PROTOTYPE_BASED
        }
        
        logger.info(f"原型网络适应完成，性能: {performance:.4f}")
        return adaptation_result
        
    def evaluate_adaptation(self, task: Task, adapted_model: Any) -> float:
        """评估适应效果"""
        if isinstance(adapted_model, dict) and 'prototypes' in adapted_model:
            return self._evaluate_prototypes(task.query_set, adapted_model['prototypes'])
        return 0.0

class MemoryAugmentedNetwork(MetaLearningAlgorithm):
    """记忆增强网络"""
    
    def __init__(self, memory_size: int = 1000, memory_dim: int = 128):
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.external_memory = {}
        self.memory_usage_count = defaultdict(int)
        
    def meta_train(self, tasks: List[Task]) -> Dict[str, Any]:
        """记忆增强网络元训练"""
        logger.info(f"开始记忆增强网络元训练，任务数量: {len(tasks)}")
        
        memories_stored = 0
        
        for task in tasks:
            # 从任务中提取并存储记忆
            task_memories = self._extract_task_memories(task)
            
            for memory_key, memory_value in task_memories.items():
                if len(self.external_memory) < self.memory_size:
                    self.external_memory[memory_key] = memory_value
                    memories_stored += 1
                else:
                    # 记忆已满，需要替换策略
                    self._replace_memory(memory_key, memory_value)
                    
        training_result = {
            'memories_stored': memories_stored,
            'memory_utilization': len(self.external_memory) / self.memory_size,
            'strategy': MetaLearningStrategy.MEMORY_AUGMENTED
        }
        
        logger.info(f"记忆增强网络元训练完成，存储记忆: {memories_stored}")
        return training_result
        
    def _extract_task_memories(self, task: Task) -> Dict[str, Any]:
        """从任务中提取记忆"""
        memories = {}
        
        # 从支持集中提取模式
        for i, (input_data, output_data) in enumerate(task.support_set):
            memory_key = f"{task.task_id}_pattern_{i}"
            memory_value = {
                'input_pattern': self._encode_pattern(input_data),
                'output_pattern': self._encode_pattern(output_data),
                'task_context': task.domain,
                'difficulty': task.difficulty,
                'timestamp': time.time()
            }
            memories[memory_key] = memory_value
            
        return memories
        
    def _encode_pattern(self, data: Any) -> List[float]:
        """编码数据模式"""
        if isinstance(data, str):
            # 文本模式编码
            pattern = [hash(data) % 256 / 255.0]
            pattern.extend([len(data) / 100.0])  # 长度特征
        elif isinstance(data, (int, float)):
            # 数值模式编码
            pattern = [data % 1.0, math.log(abs(data) + 1) / 10.0]
        else:
            # 默认模式
            pattern = [random.random()]
            
        # 扩展到固定维度
        while len(pattern) < self.memory_dim:
            pattern.append(0.0)
            
        return pattern[:self.memory_dim]
        
    def _replace_memory(self, new_key: str, new_value: Any):
        """记忆替换策略"""
        # 找到使用次数最少的记忆进行替换
        least_used_key = min(self.external_memory.keys(), 
                           key=lambda k: self.memory_usage_count[k])
        
        del self.external_memory[least_used_key]
        del self.memory_usage_count[least_used_key]
        
        self.external_memory[new_key] = new_value
        
    def fast_adapt(self, new_task: Task, adaptation_steps: int = 3) -> Dict[str, Any]:
        """记忆增强网络快速适应"""
        logger.info(f"记忆增强网络快速适应任务: {new_task.task_id}")
        
        # 检索相关记忆
        relevant_memories = self._retrieve_relevant_memories(new_task)
        
        # 基于记忆进行适应
        adaptation_strategy = self._generate_adaptation_strategy(relevant_memories)
        
        # 评估适应效果
        performance = self._evaluate_with_memory(new_task, relevant_memories)
        
        adaptation_result = {
            'relevant_memories': len(relevant_memories),
            'adaptation_strategy': adaptation_strategy,
            'performance': performance,
            'adaptation_steps': adaptation_steps,
            'strategy': MetaLearningStrategy.MEMORY_AUGMENTED
        }
        
        logger.info(f"记忆增强适应完成，使用记忆: {len(relevant_memories)}, 性能: {performance:.4f}")
        return adaptation_result
        
    def _retrieve_relevant_memories(self, task: Task) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        relevant_memories = []
        task_patterns = [self._encode_pattern(inp) for inp, _ in task.support_set]
        
        for memory_key, memory_value in self.external_memory.items():
            # 计算记忆与任务的相关性
            memory_pattern = memory_value['input_pattern']
            max_similarity = 0.0
            
            for task_pattern in task_patterns:
                similarity = self._compute_pattern_similarity(memory_pattern, task_pattern)
                max_similarity = max(max_similarity, similarity)
                
            # 如果相关性足够高，添加到相关记忆
            if max_similarity > 0.7:
                memory_value['similarity'] = max_similarity
                relevant_memories.append(memory_value)
                self.memory_usage_count[memory_key] += 1
                
        # 按相关性排序
        relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_memories[:10]  # 返回最相关的10个记忆
        
    def _compute_pattern_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """计算模式相似性"""
        if len(pattern1) != len(pattern2):
            return 0.0
            
        # 余弦相似性
        dot_product = sum(a * b for a, b in zip(pattern1, pattern2))
        norm1 = math.sqrt(sum(a * a for a in pattern1))
        norm2 = math.sqrt(sum(b * b for b in pattern2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _generate_adaptation_strategy(self, memories: List[Dict[str, Any]]) -> str:
        """基于记忆生成适应策略"""
        if not memories:
            return "default_strategy"
            
        # 基于记忆的难度和上下文生成策略
        avg_difficulty = sum(m.get('difficulty', 0.5) for m in memories) / len(memories)
        
        if avg_difficulty > 0.7:
            return "conservative_adaptation"
        elif avg_difficulty < 0.3:
            return "aggressive_adaptation"
        else:
            return "balanced_adaptation"
            
    def _evaluate_with_memory(self, task: Task, memories: List[Dict[str, Any]]) -> float:
        """基于记忆评估性能"""
        if not memories:
            return 0.5  # 默认性能
            
        # 基于记忆相关性计算性能
        avg_similarity = sum(m.get('similarity', 0.0) for m in memories) / len(memories)
        
        # 性能与记忆相关性和任务复杂度相关
        base_performance = avg_similarity
        difficulty_penalty = task.difficulty * 0.2
        
        performance = max(0.0, base_performance - difficulty_penalty)
        return min(1.0, performance)
        
    def evaluate_adaptation(self, task: Task, adapted_model: Any) -> float:
        """评估适应效果"""
        if isinstance(adapted_model, dict) and 'performance' in adapted_model:
            return adapted_model['performance']
        return 0.5

class MetaLearningFramework:
    """元学习框架主控制器"""
    
    def __init__(self):
        self.algorithms = {
            MetaLearningStrategy.MODEL_AGNOSTIC: ModelAgnosticMetaLearning(),
            MetaLearningStrategy.PROTOTYPE_BASED: PrototypeNetwork(),
            MetaLearningStrategy.MEMORY_AUGMENTED: MemoryAugmentedNetwork()
        }
        
        self.task_similarity_measurer = TaskSimilarityMeasurer()
        self.meta_knowledge_base = {}
        self.learning_experiences = []
        self.task_history = []
        
        # 性能统计
        self.performance_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'avg_adaptation_performance': 0.0,
            'strategy_performance': defaultdict(list),
            'adaptation_times': defaultdict(list)
        }
        
    def meta_train_all_algorithms(self, training_tasks: List[Task]) -> Dict[str, Any]:
        """对所有算法进行元训练"""
        logger.info(f"开始元训练所有算法，训练任务数: {len(training_tasks)}")
        
        training_results = {}
        
        for strategy, algorithm in self.algorithms.items():
            start_time = time.time()
            
            try:
                result = algorithm.meta_train(training_tasks)
                result['training_time'] = time.time() - start_time
                training_results[strategy.value] = result
                
                logger.info(f"{strategy.value}元训练完成，耗时: {result['training_time']:.2f}秒")
                
            except Exception as e:
                logger.error(f"{strategy.value}元训练失败: {str(e)}")
                training_results[strategy.value] = {'error': str(e)}
                
        return training_results
        
    def select_best_strategy(self, new_task: Task) -> MetaLearningStrategy:
        """选择最佳的元学习策略"""
        if not self.learning_experiences:
            # 没有经验时，默认使用MAML
            return MetaLearningStrategy.MODEL_AGNOSTIC
            
        # 找到相似任务的历史经验
        similar_tasks = self.task_similarity_measurer.find_similar_tasks(
            new_task, self.task_history, top_k=5
        )
        
        if not similar_tasks:
            return MetaLearningStrategy.MODEL_AGNOSTIC
            
        # 统计各策略在相似任务上的表现
        strategy_scores = defaultdict(list)
        
        for experience in self.learning_experiences:
            for similar_task, similarity in similar_tasks:
                if experience.task_id == similar_task.task_id:
                    weighted_performance = experience.performance * similarity
                    strategy_scores[experience.strategy_used].append(weighted_performance)
                    
        # 选择平均表现最好的策略
        best_strategy = MetaLearningStrategy.MODEL_AGNOSTIC
        best_score = 0.0
        
        for strategy, scores in strategy_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = strategy
                    
        logger.info(f"为任务 {new_task.task_id} 选择策略: {best_strategy.value} (预期性能: {best_score:.3f})")
        return best_strategy
        
    def fast_adapt_to_new_task(self, new_task: Task, strategy: Optional[MetaLearningStrategy] = None,
                              adaptation_steps: int = 5) -> Dict[str, Any]:
        """快速适应新任务"""
        if strategy is None:
            strategy = self.select_best_strategy(new_task)
            
        logger.info(f"开始快速适应任务: {new_task.task_id}，使用策略: {strategy.value}")
        
        start_time = time.time()
        
        try:
            algorithm = self.algorithms[strategy]
            adaptation_result = algorithm.fast_adapt(new_task, adaptation_steps)
            
            adaptation_time = time.time() - start_time
            adaptation_result['adaptation_time'] = adaptation_time
            adaptation_result['selected_strategy'] = strategy.value
            
            # 记录学习经验
            experience = LearningExperience(
                task_id=new_task.task_id,
                strategy_used=strategy,
                performance=adaptation_result.get('performance', adaptation_result.get('final_performance', 0.5)),
                adaptation_steps=adaptation_steps,
                learning_curve=adaptation_result.get('adaptation_curve', []),
                insights=self._extract_insights(adaptation_result)
            )
            
            self.learning_experiences.append(experience)
            self.task_history.append(new_task)
            
            # 更新性能指标
            self._update_performance_metrics(strategy, experience, adaptation_time)
            
            logger.info(f"快速适应完成，性能: {experience.performance:.4f}，耗时: {adaptation_time:.3f}秒")
            
            return adaptation_result
            
        except Exception as e:
            logger.error(f"快速适应失败: {str(e)}")
            return {'error': str(e), 'strategy': strategy.value}
            
    def _extract_insights(self, adaptation_result: Dict[str, Any]) -> List[str]:
        """从适应结果中提取洞察"""
        insights = []
        
        performance = adaptation_result.get('performance', adaptation_result.get('final_performance', 0))
        
        if performance > 0.8:
            insights.append("高性能适应，策略选择正确")
        elif performance < 0.4:
            insights.append("适应性能较低，可能需要更多适应步骤或不同策略")
            
        adaptation_curve = adaptation_result.get('adaptation_curve', [])
        if len(adaptation_curve) > 1:
            if adaptation_curve[-1] < adaptation_curve[0]:
                insights.append("学习曲线显示持续改进")
            else:
                insights.append("可能存在过拟合或收敛问题")
                
        return insights
        
    def _update_performance_metrics(self, strategy: MetaLearningStrategy, 
                                  experience: LearningExperience, adaptation_time: float):
        """更新性能指标"""
        self.performance_metrics['total_adaptations'] += 1
        
        if experience.performance > 0.6:  # 认为性能>0.6为成功
            self.performance_metrics['successful_adaptations'] += 1
            
        # 更新平均性能
        total_adaptations = self.performance_metrics['total_adaptations']
        current_avg = self.performance_metrics['avg_adaptation_performance']
        new_avg = (current_avg * (total_adaptations - 1) + experience.performance) / total_adaptations
        self.performance_metrics['avg_adaptation_performance'] = new_avg
        
        # 记录策略特定性能
        self.performance_metrics['strategy_performance'][strategy.value].append(experience.performance)
        self.performance_metrics['adaptation_times'][strategy.value].append(adaptation_time)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        success_rate = (
            self.performance_metrics['successful_adaptations'] / 
            max(self.performance_metrics['total_adaptations'], 1)
        )
        
        strategy_stats = {}
        for strategy, performances in self.performance_metrics['strategy_performance'].items():
            if performances:
                strategy_stats[strategy] = {
                    'avg_performance': sum(performances) / len(performances),
                    'best_performance': max(performances),
                    'usage_count': len(performances),
                    'avg_time': sum(self.performance_metrics['adaptation_times'][strategy]) / len(performances)
                }
                
        return {
            "总适应次数": self.performance_metrics['total_adaptations'],
            "成功适应次数": self.performance_metrics['successful_adaptations'],
            "成功率": f"{success_rate:.2%}",
            "平均适应性能": f"{self.performance_metrics['avg_adaptation_performance']:.3f}",
            "策略统计": strategy_stats,
            "学习经验数": len(self.learning_experiences),
            "任务历史数": len(self.task_history)
        }
        
    def generate_meta_insights(self) -> List[str]:
        """生成元学习洞察"""
        insights = []
        
        if len(self.learning_experiences) < 5:
            insights.append("经验数据不足，建议进行更多元训练")
            return insights
            
        # 分析策略偏好
        strategy_usage = defaultdict(int)
        for exp in self.learning_experiences:
            strategy_usage[exp.strategy_used.value] += 1
            
        most_used_strategy = max(strategy_usage.items(), key=lambda x: x[1])
        insights.append(f"最常用策略: {most_used_strategy[0]} (使用{most_used_strategy[1]}次)")
        
        # 分析性能趋势
        recent_performances = [exp.performance for exp in self.learning_experiences[-10:]]
        if len(recent_performances) >= 5:
            trend = sum(recent_performances[-3:]) / 3 - sum(recent_performances[:3]) / 3
            if trend > 0.05:
                insights.append("近期适应性能呈上升趋势")
            elif trend < -0.05:
                insights.append("近期适应性能有所下降，建议分析原因")
                
        # 分析任务特性
        task_types = [task.task_type for task in self.task_history]
        if task_types:
            most_common_type = max(set(task_types), key=task_types.count)
            insights.append(f"最常处理的任务类型: {most_common_type.value}")
            
        return insights

# 示例使用和测试
async def demonstrate_meta_learning():
    """演示元学习框架功能"""
    print("🧠 自主进化Agent - 第3轮提升：元学习框架")
    print("=" * 60)
    
    # 创建元学习框架
    meta_framework = MetaLearningFramework()
    
    # 生成训练任务
    print("\n📚 生成元训练任务")
    training_tasks = []
    
    for i in range(5):
        task = Task(
            task_id=f"train_task_{i}",
            task_type=TaskType.CLASSIFICATION,
            domain=f"domain_{i % 3}",
            description=f"分类任务{i}：识别数据模式",
            support_set=[(f"input_{j}", f"label_{j % 2}") for j in range(10)],
            query_set=[(f"query_{j}", f"label_{j % 2}") for j in range(5)],
            difficulty=random.uniform(0.3, 0.8)
        )
        training_tasks.append(task)
        
    # 元训练
    print("\n🔧 开始元训练")
    training_results = meta_framework.meta_train_all_algorithms(training_tasks)
    
    for strategy, result in training_results.items():
        if 'error' not in result:
            print(f"  {strategy}: 训练时间 {result.get('training_time', 0):.2f}秒")
        else:
            print(f"  {strategy}: 训练失败 - {result['error']}")
            
    # 快速适应新任务
    print("\n⚡ 快速适应新任务")
    new_tasks = [
        Task(
            task_id="new_task_1",
            task_type=TaskType.CLASSIFICATION,
            domain="domain_1",
            description="新的分类任务：文本情感分析",
            support_set=[("正面文本", "positive"), ("负面文本", "negative")],
            query_set=[("测试文本", "positive")],
            difficulty=0.6
        ),
        Task(
            task_id="new_task_2",
            task_type=TaskType.REGRESSION,
            domain="domain_2",
            description="回归任务：数值预测",
            support_set=[(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)],
            query_set=[(4.0, 8.0)],
            difficulty=0.4
        )
    ]
    
    for new_task in new_tasks:
        print(f"\n🎯 适应任务: {new_task.task_id}")
        adaptation_result = meta_framework.fast_adapt_to_new_task(new_task)
        
        if 'error' not in adaptation_result:
            performance = adaptation_result.get('performance', adaptation_result.get('final_performance', 0))
            strategy = adaptation_result.get('selected_strategy', 'unknown')
            time_cost = adaptation_result.get('adaptation_time', 0)
            
            print(f"  策略: {strategy}")
            print(f"  性能: {performance:.3f}")
            print(f"  耗时: {time_cost:.3f}秒")
        else:
            print(f"  适应失败: {adaptation_result['error']}")
            
    # 性能报告
    print("\n📊 元学习性能报告")
    report = meta_framework.get_performance_report()
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
            
    # 元学习洞察
    print("\n💡 元学习洞察")
    insights = meta_framework.generate_meta_insights()
    for insight in insights:
        print(f"  • {insight}")
        
    print("\n✅ 第3轮提升完成！元学习框架已成功部署")

if __name__ == "__main__":
    asyncio.run(demonstrate_meta_learning())