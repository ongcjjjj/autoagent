"""
高级学习与适应引擎
实现在线学习、强化学习、元学习、知识蒸馏
"""
import json
import time
import math
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class LearningExample:
    """学习样本"""
    input_data: Dict[str, Any]
    expected_output: Any
    actual_output: Any
    feedback_score: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    difficulty_level: float = 0.5
    learning_category: str = "general"

@dataclass
class LearningPattern:
    """学习模式"""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    last_updated: float = field(default_factory=time.time)
    effectiveness_score: float = 0.5

@dataclass
class MetaLearningState:
    """元学习状态"""
    learning_strategy: str
    adaptation_rate: float
    exploration_rate: float
    exploitation_rate: float
    current_task_context: Dict[str, Any] = field(default_factory=dict)
    meta_knowledge: Dict[str, Any] = field(default_factory=dict)

class LearningStrategy(ABC):
    """学习策略抽象基类"""
    
    @abstractmethod
    def learn_from_example(self, example: LearningExample) -> Dict[str, Any]:
        """从样本中学习"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> Any:
        """预测"""
        pass
    
    @abstractmethod
    def get_confidence(self, input_data: Dict[str, Any]) -> float:
        """获取预测置信度"""
        pass

class OnlineLearningStrategy(LearningStrategy):
    """在线学习策略"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.feature_weights = defaultdict(float)
        self.bias = 0.0
        self.sample_count = 0
        self.feature_importance = defaultdict(float)
        
    def learn_from_example(self, example: LearningExample) -> Dict[str, Any]:
        """从样本中学习 - 在线梯度下降"""
        self.sample_count += 1
        
        # 提取特征
        features = self._extract_features(example.input_data)
        
        # 计算预测
        prediction = self._predict_score(features)
        
        # 计算误差
        error = example.feedback_score - prediction
        
        # 更新权重
        for feature, value in features.items():
            self.feature_weights[feature] += self.learning_rate * error * value
            self.feature_importance[feature] = (
                self.feature_importance[feature] * 0.9 + 
                abs(self.learning_rate * error * value) * 0.1
            )
        
        self.bias += self.learning_rate * error
        
        # 动态调整学习率
        self.learning_rate *= 0.999  # 逐渐降低学习率
        
        return {
            "error": abs(error),
            "prediction": prediction,
            "updated_weights": len(features),
            "total_features": len(self.feature_weights)
        }
    
    def predict(self, input_data: Dict[str, Any]) -> float:
        """预测得分"""
        features = self._extract_features(input_data)
        return self._predict_score(features)
    
    def get_confidence(self, input_data: Dict[str, Any]) -> float:
        """获取预测置信度"""
        features = self._extract_features(input_data)
        
        # 基于特征重要性计算置信度
        total_importance = 0.0
        covered_importance = 0.0
        
        for feature, importance in self.feature_importance.items():
            total_importance += importance
            if feature in features:
                covered_importance += importance
        
        if total_importance == 0:
            return 0.5
        
        confidence = covered_importance / total_importance
        
        # 考虑样本数量的影响
        sample_confidence = min(self.sample_count / 100.0, 1.0)
        
        return confidence * sample_confidence
    
    def _extract_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """提取特征"""
        features = {}
        
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                features[f"num_{key}"] = float(value)
            elif isinstance(value, str):
                features[f"str_{key}_len"] = len(value)
                # 简单的文本特征
                features[f"str_{key}_words"] = len(value.split())
            elif isinstance(value, bool):
                features[f"bool_{key}"] = 1.0 if value else 0.0
            elif isinstance(value, list):
                features[f"list_{key}_len"] = len(value)
        
        return features
    
    def _predict_score(self, features: Dict[str, float]) -> float:
        """计算预测得分"""
        score = self.bias
        for feature, value in features.items():
            score += self.feature_weights[feature] * value
        
        # 应用sigmoid激活函数
        return 1.0 / (1.0 + math.exp(-score))

class ReinforcementLearningStrategy(LearningStrategy):
    """强化学习策略"""
    
    def __init__(self, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.9):
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.reward_history = deque(maxlen=1000)
        
    def learn_from_example(self, example: LearningExample) -> Dict[str, Any]:
        """Q-learning更新"""
        state = self._extract_state(example.input_data)
        action = self._extract_action(example.expected_output)
        reward = example.feedback_score
        next_state = self._extract_state(example.context.get("next_state", {}))
        
        # Q-learning更新公式
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        self.state_action_counts[state][action] += 1
        self.reward_history.append(reward)
        
        # 动态调整探索率
        self._update_exploration_rate()
        
        return {
            "q_value_updated": new_q,
            "state": state,
            "action": action,
            "reward": reward,
            "exploration_rate": self.epsilon
        }
    
    def predict(self, input_data: Dict[str, Any]) -> str:
        """选择最佳动作"""
        state = self._extract_state(input_data)
        
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            available_actions = list(self.q_table[state].keys()) if state in self.q_table else ["default"]
            return random.choice(available_actions) if available_actions else "default"
        else:
            # 利用：选择最佳动作
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state].keys(), key=lambda a: self.q_table[state][a])
            return "default"
    
    def get_confidence(self, input_data: Dict[str, Any]) -> float:
        """获取动作选择置信度"""
        state = self._extract_state(input_data)
        
        if state not in self.q_table or not self.q_table[state]:
            return 0.1
        
        q_values = list(self.q_table[state].values())
        if len(q_values) < 2:
            return 0.5
        
        # 基于Q值差异计算置信度
        max_q = max(q_values)
        second_max_q = sorted(q_values)[-2]
        
        confidence = (max_q - second_max_q + 1.0) / 2.0
        return min(max(confidence, 0.0), 1.0)
    
    def _extract_state(self, data: Dict[str, Any]) -> str:
        """提取状态表示"""
        # 简化的状态表示
        state_features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                state_features.append(f"{key}_{int(value*10)}")
            elif isinstance(value, str):
                state_features.append(f"{key}_{len(value)//10}")
            elif isinstance(value, bool):
                state_features.append(f"{key}_{value}")
        
        return "_".join(sorted(state_features)[:5])  # 限制状态维度
    
    def _extract_action(self, output: Any) -> str:
        """提取动作表示"""
        if isinstance(output, str):
            return output[:20]  # 限制动作长度
        return str(output)
    
    def _update_exploration_rate(self):
        """更新探索率"""
        # 基于最近奖励的方差调整探索率
        if len(self.reward_history) > 10:
            recent_rewards = list(self.reward_history)[-10:]
            variance = np.var(recent_rewards) if len(recent_rewards) > 1 else 1.0
            
            # 高方差时增加探索，低方差时减少探索
            target_epsilon = min(0.3, max(0.01, variance))
            self.epsilon = self.epsilon * 0.95 + target_epsilon * 0.05

class MetaLearningStrategy(LearningStrategy):
    """元学习策略"""
    
    def __init__(self):
        self.task_models = {}
        self.meta_model = OnlineLearningStrategy(learning_rate=0.05)
        self.task_similarities = defaultdict(lambda: defaultdict(float))
        self.adaptation_history = deque(maxlen=100)
        
    def learn_from_example(self, example: LearningExample) -> Dict[str, Any]:
        """元学习更新"""
        task_id = example.learning_category
        
        # 为新任务创建模型
        if task_id not in self.task_models:
            self.task_models[task_id] = OnlineLearningStrategy(learning_rate=0.1)
        
        # 在任务特定模型上学习
        task_result = self.task_models[task_id].learn_from_example(example)
        
        # 在元模型上学习任务特征
        meta_features = self._extract_meta_features(example)
        meta_example = LearningExample(
            input_data=meta_features,
            expected_output=example.feedback_score,
            actual_output=task_result.get("prediction", 0.5),
            feedback_score=example.feedback_score,
            learning_category="meta"
        )
        meta_result = self.meta_model.learn_from_example(meta_example)
        
        # 更新任务相似性
        self._update_task_similarities(task_id, example)
        
        # 记录适应历史
        self.adaptation_history.append({
            "task_id": task_id,
            "meta_features": meta_features,
            "adaptation_success": example.feedback_score > 0.6,
            "timestamp": time.time()
        })
        
        return {
            "task_result": task_result,
            "meta_result": meta_result,
            "task_count": len(self.task_models),
            "adaptation_rate": self._calculate_adaptation_rate()
        }
    
    def predict(self, input_data: Dict[str, Any]) -> Any:
        """元学习预测"""
        task_id = input_data.get("learning_category", "general")
        
        if task_id in self.task_models:
            # 使用任务特定模型
            return self.task_models[task_id].predict(input_data)
        else:
            # 寻找最相似的任务
            similar_task = self._find_most_similar_task(input_data)
            if similar_task and similar_task in self.task_models:
                # 快速适应：从相似任务开始
                new_model = OnlineLearningStrategy(learning_rate=0.2)
                new_model.feature_weights = self.task_models[similar_task].feature_weights.copy()
                new_model.bias = self.task_models[similar_task].bias
                self.task_models[task_id] = new_model
                
                return new_model.predict(input_data)
            else:
                # 使用元模型预测
                meta_features = self._extract_meta_features_from_input(input_data)
                return self.meta_model.predict(meta_features)
    
    def get_confidence(self, input_data: Dict[str, Any]) -> float:
        """获取元学习置信度"""
        task_id = input_data.get("learning_category", "general")
        
        if task_id in self.task_models:
            task_confidence = self.task_models[task_id].get_confidence(input_data)
            # 基于任务经验调整置信度
            task_experience = self.task_models[task_id].sample_count
            experience_factor = min(task_experience / 50.0, 1.0)
            return task_confidence * experience_factor
        else:
            # 基于任务相似性的置信度
            similarity_score = self._calculate_task_similarity_confidence(input_data)
            return similarity_score * 0.7  # 降低未见任务的置信度
    
    def _extract_meta_features(self, example: LearningExample) -> Dict[str, Any]:
        """提取元特征"""
        return {
            "input_complexity": len(str(example.input_data)),
            "output_type": type(example.expected_output).__name__,
            "context_size": len(example.context),
            "difficulty": example.difficulty_level,
            "category": example.learning_category,
            "feedback_score": example.feedback_score
        }
    
    def _extract_meta_features_from_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """从输入数据提取元特征"""
        return {
            "input_complexity": len(str(input_data)),
            "data_types": "_".join([type(v).__name__ for v in input_data.values()]),
            "num_features": len(input_data),
            "category": input_data.get("learning_category", "general")
        }
    
    def _update_task_similarities(self, task_id: str, example: LearningExample):
        """更新任务相似性"""
        meta_features = self._extract_meta_features(example)
        
        for other_task_id in self.task_models:
            if other_task_id != task_id:
                # 简单的特征相似性计算
                similarity = self._calculate_feature_similarity(task_id, other_task_id, meta_features)
                self.task_similarities[task_id][other_task_id] = (
                    self.task_similarities[task_id][other_task_id] * 0.9 + similarity * 0.1
                )
    
    def _calculate_feature_similarity(self, task1: str, task2: str, features: Dict[str, Any]) -> float:
        """计算特征相似性"""
        # 简化的相似性计算
        return 0.5 + random.uniform(-0.2, 0.2)  # 占位符实现
    
    def _find_most_similar_task(self, input_data: Dict[str, Any]) -> Optional[str]:
        """找到最相似的任务"""
        if not self.task_similarities:
            return None
        
        # 基于输入特征找到最相似的任务
        best_task = None
        best_similarity = 0.0
        
        for task_id in self.task_models:
            # 简化的相似性计算
            similarity = random.uniform(0.3, 0.8)  # 占位符实现
            if similarity > best_similarity:
                best_similarity = similarity
                best_task = task_id
        
        return best_task if best_similarity > 0.5 else None
    
    def _calculate_adaptation_rate(self) -> float:
        """计算适应率"""
        if len(self.adaptation_history) < 10:
            return 0.5
        
        recent_adaptations = list(self.adaptation_history)[-10:]
        success_count = sum(1 for adapt in recent_adaptations if adapt["adaptation_success"])
        
        return success_count / len(recent_adaptations)
    
    def _calculate_task_similarity_confidence(self, input_data: Dict[str, Any]) -> float:
        """计算基于任务相似性的置信度"""
        if not self.task_similarities:
            return 0.3
        
        # 简化实现
        return 0.6

class AdaptiveLearningEngine:
    """自适应学习引擎主类"""
    
    def __init__(self):
        self.strategies = {
            "online": OnlineLearningStrategy(),
            "reinforcement": ReinforcementLearningStrategy(),
            "meta": MetaLearningStrategy()
        }
        
        self.current_strategy = "online"
        self.strategy_performance = defaultdict(lambda: {"success_rate": 0.5, "usage_count": 0})
        self.learning_examples = deque(maxlen=1000)
        self.learning_patterns = {}
        
        self.meta_state = MetaLearningState(
            learning_strategy=self.current_strategy,
            adaptation_rate=0.1,
            exploration_rate=0.2,
            exploitation_rate=0.8
        )
        
        self.performance_monitor = {
            "total_examples": 0,
            "average_performance": 0.5,
            "learning_trends": deque(maxlen=50),
            "strategy_switches": 0
        }
    
    async def learn_from_interaction(
        self, 
        input_data: Dict[str, Any], 
        expected_output: Any, 
        actual_output: Any, 
        feedback_score: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """从交互中学习"""
        if context is None:
            context = {}
        
        # 创建学习样本
        example = LearningExample(
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
            feedback_score=feedback_score,
            context=context,
            difficulty_level=self._assess_difficulty(input_data, expected_output),
            learning_category=context.get("category", "general")
        )
        
        self.learning_examples.append(example)
        
        # 选择学习策略
        strategy_name = self._select_learning_strategy(example)
        strategy = self.strategies[strategy_name]
        
        # 执行学习
        learning_result = strategy.learn_from_example(example)
        
        # 更新策略性能
        self._update_strategy_performance(strategy_name, feedback_score)
        
        # 检测学习模式
        pattern = self._detect_learning_pattern(example)
        if pattern:
            self._update_learning_pattern(pattern)
        
        # 更新性能监控
        self._update_performance_monitor(feedback_score)
        
        # 元学习：学习如何学习
        meta_learning_result = await self._meta_learn(example, learning_result)
        
        return {
            "strategy_used": strategy_name,
            "learning_result": learning_result,
            "meta_learning_result": meta_learning_result,
            "detected_pattern": pattern.pattern_id if pattern else None,
            "confidence": strategy.get_confidence(input_data),
            "performance_summary": self._get_performance_summary()
        }
    
    def _select_learning_strategy(self, example: LearningExample) -> str:
        """选择学习策略"""
        # 基于任务特征和历史性能选择策略
        
        # 如果是序列决策任务，优先使用强化学习
        if "action" in example.input_data or "state" in example.input_data:
            return "reinforcement"
        
        # 如果是多任务场景，使用元学习
        if len(set(ex.learning_category for ex in list(self.learning_examples)[-10:])) > 2:
            return "meta"
        
        # 基于策略性能选择
        best_strategy = max(
            self.strategy_performance.keys(),
            key=lambda s: self.strategy_performance[s]["success_rate"],
            default="online"
        )
        
        # 探索vs利用
        if random.random() < self.meta_state.exploration_rate:
            # 探索：选择随机策略
            return random.choice(list(self.strategies.keys()))
        else:
            # 利用：选择最佳策略
            return best_strategy
    
    def _assess_difficulty(self, input_data: Dict[str, Any], expected_output: Any) -> float:
        """评估样本难度"""
        difficulty = 0.5
        
        # 基于输入复杂度
        input_complexity = len(str(input_data)) / 1000.0
        difficulty += min(input_complexity, 0.3)
        
        # 基于输出复杂度
        output_complexity = len(str(expected_output)) / 1000.0
        difficulty += min(output_complexity, 0.2)
        
        return min(difficulty, 1.0)
    
    def _detect_learning_pattern(self, example: LearningExample) -> Optional[LearningPattern]:
        """检测学习模式"""
        # 简化的模式检测
        pattern_conditions = {
            "input_type": type(example.input_data).__name__,
            "output_type": type(example.expected_output).__name__,
            "category": example.learning_category,
            "difficulty_range": "high" if example.difficulty_level > 0.7 else "low"
        }
        
        pattern_id = "_".join([f"{k}_{v}" for k, v in pattern_conditions.items()])
        
        if pattern_id not in self.learning_patterns:
            return LearningPattern(
                pattern_id=pattern_id,
                pattern_type="interaction",
                conditions=pattern_conditions,
                actions=[self.current_strategy]
            )
        
        return self.learning_patterns[pattern_id]
    
    def _update_learning_pattern(self, pattern: LearningPattern):
        """更新学习模式"""
        if pattern.pattern_id in self.learning_patterns:
            existing_pattern = self.learning_patterns[pattern.pattern_id]
            existing_pattern.usage_count += 1
            existing_pattern.last_updated = time.time()
        else:
            self.learning_patterns[pattern.pattern_id] = pattern
    
    def _update_strategy_performance(self, strategy_name: str, feedback_score: float):
        """更新策略性能"""
        perf = self.strategy_performance[strategy_name]
        perf["usage_count"] += 1
        
        # 移动平均
        alpha = 0.1
        perf["success_rate"] = (
            perf["success_rate"] * (1 - alpha) + 
            (1.0 if feedback_score > 0.6 else 0.0) * alpha
        )
    
    def _update_performance_monitor(self, feedback_score: float):
        """更新性能监控"""
        self.performance_monitor["total_examples"] += 1
        self.performance_monitor["learning_trends"].append(feedback_score)
        
        # 更新平均性能
        recent_trends = list(self.performance_monitor["learning_trends"])
        if recent_trends:
            self.performance_monitor["average_performance"] = sum(recent_trends) / len(recent_trends)
    
    async def _meta_learn(self, example: LearningExample, learning_result: Dict[str, Any]) -> Dict[str, Any]:
        """元学习：学习如何学习"""
        # 分析学习效果
        learning_effectiveness = learning_result.get("error", 1.0)
        
        # 调整元参数
        if learning_effectiveness < 0.1:  # 学习效果好
            self.meta_state.adaptation_rate = min(self.meta_state.adaptation_rate * 1.05, 0.5)
            self.meta_state.exploration_rate = max(self.meta_state.exploration_rate * 0.95, 0.1)
        else:  # 学习效果差
            self.meta_state.exploration_rate = min(self.meta_state.exploration_rate * 1.1, 0.4)
        
        self.meta_state.exploitation_rate = 1.0 - self.meta_state.exploration_rate
        
        return {
            "adaptation_rate": self.meta_state.adaptation_rate,
            "exploration_rate": self.meta_state.exploration_rate,
            "learning_effectiveness": learning_effectiveness,
            "meta_knowledge_size": len(self.meta_state.meta_knowledge)
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "total_examples": self.performance_monitor["total_examples"],
            "average_performance": self.performance_monitor["average_performance"],
            "strategy_performance": dict(self.strategy_performance),
            "current_strategy": self.current_strategy,
            "learning_patterns_count": len(self.learning_patterns),
            "recent_trend": "improving" if len(self.performance_monitor["learning_trends"]) > 5 and 
                           list(self.performance_monitor["learning_trends"])[-5:] > 
                           list(self.performance_monitor["learning_trends"])[-10:-5] else "stable"
        }
    
    def predict_with_adaptation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """自适应预测"""
        predictions = {}
        confidences = {}
        
        # 使用所有策略进行预测
        for strategy_name, strategy in self.strategies.items():
            try:
                prediction = strategy.predict(input_data)
                confidence = strategy.get_confidence(input_data)
                
                predictions[strategy_name] = prediction
                confidences[strategy_name] = confidence
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} prediction failed: {e}")
                predictions[strategy_name] = None
                confidences[strategy_name] = 0.0
        
        # 选择最佳预测
        best_strategy = max(confidences.keys(), key=lambda s: confidences[s])
        
        return {
            "best_prediction": predictions[best_strategy],
            "best_strategy": best_strategy,
            "all_predictions": predictions,
            "all_confidences": confidences,
            "ensemble_confidence": sum(confidences.values()) / len(confidences)
        }