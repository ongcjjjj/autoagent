"""
自适应进化引擎模块
集成多种进化算法、智能策略选择、多目标优化等功能
"""
import json
import time
import random
import statistics
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict

# 导入基础模块（处理可能的导入错误）
try:
    from genetic_evolution import (
        GeneticEvolutionManager, DifferentialEvolution, 
        MemeticAlgorithm, CoevolutionaryAlgorithm,
        Individual, Population
    )
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False

try:
    from enhanced_memory import EnhancedMemoryManager, EnhancedMemory
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    ENHANCED_MEMORY_AVAILABLE = False

from evolution import EvolutionEngine, EvolutionMetrics, EvolutionRecord
from memory import MemoryManager, Memory

class EvolutionStrategy(Enum):
    """进化策略枚举"""
    EXPLORATION = "exploration"  # 探索
    EXPLOITATION = "exploitation"  # 开发
    BALANCED = "balanced"  # 平衡
    ADAPTIVE = "adaptive"  # 自适应

class OptimizationObjective(Enum):
    """优化目标枚举"""
    SUCCESS_RATE = "success_rate"
    RESPONSE_QUALITY = "response_quality"
    LEARNING_EFFICIENCY = "learning_efficiency"
    ADAPTATION_SPEED = "adaptation_speed"
    USER_SATISFACTION = "user_satisfaction"

@dataclass
class AdaptiveParameters:
    """自适应参数"""
    learning_rate: float = 0.1
    mutation_rate: float = 0.1
    selection_pressure: float = 0.8
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.01
    exploration_rate: float = 0.2
    temperature: float = 1.0  # 用于模拟退火
    
    def to_genes(self) -> List[float]:
        """转换为基因表示"""
        return [
            self.learning_rate, self.mutation_rate, self.selection_pressure,
            self.diversity_threshold, self.convergence_threshold,
            self.exploration_rate, self.temperature
        ]
    
    @classmethod
    def from_genes(cls, genes: List[float]) -> 'AdaptiveParameters':
        """从基因创建参数"""
        if len(genes) != 7:
            return cls()
        
        return cls(
            learning_rate=max(0.01, min(1.0, genes[0])),
            mutation_rate=max(0.01, min(1.0, genes[1])),
            selection_pressure=max(0.1, min(1.0, genes[2])),
            diversity_threshold=max(0.1, min(1.0, genes[3])),
            convergence_threshold=max(0.001, min(0.1, genes[4])),
            exploration_rate=max(0.1, min(1.0, genes[5])),
            temperature=max(0.1, min(10.0, genes[6]))
        )

@dataclass
class MultiObjectiveResult:
    """多目标优化结果"""
    pareto_front: List[Dict[str, float]]  # 帕累托前沿
    hypervolume: float  # 超体积指标
    diversity_metric: float  # 多样性指标
    convergence_metric: float  # 收敛性指标
    dominated_solutions: int  # 被支配解的数量

@dataclass
class StrategyPerformance:
    """策略性能记录"""
    strategy: EvolutionStrategy
    success_count: int = 0
    total_attempts: int = 0
    avg_improvement: float = 0.0
    recent_performance: deque = None
    confidence: float = 0.5
    
    def __post_init__(self):
        if self.recent_performance is None:
            self.recent_performance = deque(maxlen=20)
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_attempts)
    
    def update_performance(self, improvement: float):
        """更新性能记录"""
        self.total_attempts += 1
        if improvement > 0:
            self.success_count += 1
        
        self.recent_performance.append(improvement)
        
        # 更新平均改进
        if len(self.recent_performance) > 0:
            self.avg_improvement = statistics.mean(self.recent_performance)
        
        # 更新置信度
        if len(self.recent_performance) >= 5:
            variance = statistics.variance(self.recent_performance)
            self.confidence = max(0.1, min(1.0, 1.0 / (1.0 + variance)))

class AdaptiveEvolutionEngine:
    """自适应进化引擎"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.base_evolution_engine = EvolutionEngine(memory_manager)
        
        # 策略管理
        self.strategy_performances: Dict[EvolutionStrategy, StrategyPerformance] = {
            strategy: StrategyPerformance(strategy) for strategy in EvolutionStrategy
        }
        self.current_strategy = EvolutionStrategy.BALANCED
        self.strategy_switch_threshold = 0.1
        
        # 自适应参数
        self.adaptive_params = AdaptiveParameters()
        self.param_history = []
        
        # 多目标优化
        self.optimization_objectives = [
            OptimizationObjective.SUCCESS_RATE,
            OptimizationObjective.RESPONSE_QUALITY,
            OptimizationObjective.LEARNING_EFFICIENCY
        ]
        self.pareto_archive = []
        
        # 环境适应
        self.environment_changes = []
        self.adaptation_triggers = {
            "performance_drop": 0.2,
            "diversity_loss": 0.3,
            "stagnation_period": 50
        }
        
        # 遗传进化管理器
        if GENETIC_AVAILABLE:
            self.genetic_manager = GeneticEvolutionManager()
        else:
            self.genetic_manager = None
        
        # 增强记忆管理器
        if ENHANCED_MEMORY_AVAILABLE:
            self.enhanced_memory = EnhancedMemoryManager()
        else:
            self.enhanced_memory = None
        
        # 性能跟踪
        self.performance_history = deque(maxlen=100)
        self.stagnation_counter = 0
        self.last_significant_improvement = time.time()
        
        self.load_adaptive_data()
    
    def evolve_with_strategy(self, interaction_data: Dict[str, Any]) -> EvolutionRecord:
        """使用当前策略进行进化"""
        # 评估当前表现
        performance_score = self.base_evolution_engine.evaluate_performance(interaction_data)
        self.performance_history.append(performance_score)
        
        # 检测环境变化
        environment_change = self._detect_environment_change()
        if environment_change:
            self._adapt_to_environment_change(environment_change)
        
        # 选择最优策略
        self._update_strategy_selection()
        
        # 根据策略调整参数
        self._adjust_parameters_by_strategy()
        
        # 执行进化
        evolution_record = self._execute_evolution_with_adaptive_params(interaction_data)
        
        # 更新策略性能
        self._update_strategy_performance(evolution_record)
        
        # 多目标优化
        self._update_pareto_archive(evolution_record)
        
        return evolution_record
    
    def _detect_environment_change(self) -> Optional[str]:
        """检测环境变化"""
        if len(self.performance_history) < 20:
            return None
        
        recent_performance = list(self.performance_history)[-10:]
        older_performance = list(self.performance_history)[-20:-10]
        
        recent_avg = statistics.mean(recent_performance)
        older_avg = statistics.mean(older_performance)
        
        # 性能显著下降
        if recent_avg < older_avg - self.adaptation_triggers["performance_drop"]:
            return "performance_drop"
        
        # 多样性丢失（方差过小）
        if len(recent_performance) > 1:
            recent_variance = statistics.variance(recent_performance)
            if recent_variance < self.adaptation_triggers["diversity_loss"]:
                return "diversity_loss"
        
        # 停滞期检测
        if abs(recent_avg - older_avg) < 0.01:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.adaptation_triggers["stagnation_period"]:
                return "stagnation"
        else:
            self.stagnation_counter = 0
        
        return None
    
    def _adapt_to_environment_change(self, change_type: str):
        """适应环境变化"""
        adaptation_record = {
            "timestamp": time.time(),
            "change_type": change_type,
            "old_strategy": self.current_strategy.value,
            "old_params": asdict(self.adaptive_params)
        }
        
        if change_type == "performance_drop":
            # 性能下降 -> 增加探索
            self.current_strategy = EvolutionStrategy.EXPLORATION
            self.adaptive_params.exploration_rate *= 1.5
            self.adaptive_params.mutation_rate *= 1.3
            
        elif change_type == "diversity_loss":
            # 多样性丢失 -> 增加多样性
            self.adaptive_params.mutation_rate *= 1.5
            self.adaptive_params.temperature *= 1.2
            
        elif change_type == "stagnation":
            # 停滞 -> 重新初始化部分参数
            self.adaptive_params.learning_rate *= 1.2
            self.adaptive_params.exploration_rate = 0.4
            self.stagnation_counter = 0
        
        adaptation_record["new_strategy"] = self.current_strategy.value
        adaptation_record["new_params"] = asdict(self.adaptive_params)
        
        self.environment_changes.append(adaptation_record)
        
        # 记录到记忆中
        adaptation_memory = Memory(
            content=f"环境适应: {change_type}",
            memory_type="adaptation",
            importance=0.8,
            tags=["adaptation", change_type],
            metadata=adaptation_record
        )
        self.memory_manager.add_memory(adaptation_memory)
    
    def _update_strategy_selection(self):
        """更新策略选择"""
        # 计算每个策略的期望收益
        strategy_scores = {}
        
        for strategy, performance in self.strategy_performances.items():
            if performance.total_attempts > 0:
                # 结合成功率、平均改进和置信度
                score = (performance.success_rate * 0.4 + 
                        (performance.avg_improvement + 1.0) * 0.4 + 
                        performance.confidence * 0.2)
                strategy_scores[strategy] = score
            else:
                strategy_scores[strategy] = 0.5  # 默认分数
        
        # ε-贪心策略选择
        if random.random() < self.adaptive_params.exploration_rate:
            # 探索：随机选择
            self.current_strategy = random.choice(list(EvolutionStrategy))
        else:
            # 开发：选择最优策略
            best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
            
            # 只有当新策略明显更好时才切换
            current_score = strategy_scores[self.current_strategy]
            best_score = strategy_scores[best_strategy]
            
            if best_score > current_score + self.strategy_switch_threshold:
                self.current_strategy = best_strategy
    
    def _adjust_parameters_by_strategy(self):
        """根据策略调整参数"""
        if self.current_strategy == EvolutionStrategy.EXPLORATION:
            # 探索策略：增加变异率，降低选择压力
            self.adaptive_params.mutation_rate = min(0.5, self.adaptive_params.mutation_rate * 1.2)
            self.adaptive_params.selection_pressure = max(0.3, self.adaptive_params.selection_pressure * 0.8)
            self.adaptive_params.temperature = min(5.0, self.adaptive_params.temperature * 1.3)
            
        elif self.current_strategy == EvolutionStrategy.EXPLOITATION:
            # 开发策略：降低变异率，增加选择压力
            self.adaptive_params.mutation_rate = max(0.05, self.adaptive_params.mutation_rate * 0.8)
            self.adaptive_params.selection_pressure = min(1.0, self.adaptive_params.selection_pressure * 1.2)
            self.adaptive_params.temperature = max(0.5, self.adaptive_params.temperature * 0.8)
            
        elif self.current_strategy == EvolutionStrategy.BALANCED:
            # 平衡策略：参数保持中等水平
            target_mutation = 0.15
            target_selection = 0.7
            target_temperature = 1.0
            
            self.adaptive_params.mutation_rate = (self.adaptive_params.mutation_rate + target_mutation) / 2
            self.adaptive_params.selection_pressure = (self.adaptive_params.selection_pressure + target_selection) / 2
            self.adaptive_params.temperature = (self.adaptive_params.temperature + target_temperature) / 2
            
        elif self.current_strategy == EvolutionStrategy.ADAPTIVE:
            # 自适应策略：使用遗传算法优化参数
            if self.genetic_manager and GENETIC_AVAILABLE:
                self._optimize_parameters_genetically()
    
    def _optimize_parameters_genetically(self):
        """使用遗传算法优化参数"""
        if not self.genetic_manager:
            return
        
        def parameter_fitness(genes: List[float]) -> float:
            """参数适应度函数"""
            params = AdaptiveParameters.from_genes(genes)
            
            # 基于历史性能计算适应度
            if len(self.performance_history) < 10:
                return 0.5
            
            recent_performance = list(self.performance_history)[-10:]
            performance_score = statistics.mean(recent_performance)
            
            # 考虑参数的稳定性
            stability_bonus = 1.0 - abs(params.mutation_rate - 0.15) * 0.5
            
            return performance_score * stability_bonus
        
        try:
            # 使用差分进化优化参数
            self.genetic_manager.set_algorithm("differential_evolution")
            result = self.genetic_manager.optimize(
                fitness_func=parameter_fitness,
                dimension=7,  # 7个参数
                population_size=20,
                generations=10
            )
            
            if result["best_fitness"] > 0.6:  # 只有当结果足够好时才应用
                optimized_params = AdaptiveParameters.from_genes(result["best_solution"])
                self.adaptive_params = optimized_params
                
        except Exception as e:
            print(f"遗传优化参数失败: {e}")
    
    def _execute_evolution_with_adaptive_params(self, interaction_data: Dict[str, Any]) -> EvolutionRecord:
        """使用自适应参数执行进化"""
        # 更新基础进化引擎的参数
        self.base_evolution_engine.update_performance_window(
            self.base_evolution_engine.evaluate_performance(interaction_data)
        )
        
        # 检查是否需要进化
        if self.base_evolution_engine.should_evolve():
            evolution_record = self.base_evolution_engine.execute_evolution()
            
            # 应用自适应增强
            self._apply_adaptive_enhancements(evolution_record)
            
            return evolution_record
        
        # 如果不需要进化，创建一个空的记录
        current_metrics = self.base_evolution_engine.calculate_evolution_metrics()
        return EvolutionRecord(
            version="adaptive_v" + str(len(self.base_evolution_engine.evolution_history) + 1),
            metrics=current_metrics,
            changes=[],
            improvement_areas=[]
        )
    
    def _apply_adaptive_enhancements(self, evolution_record: EvolutionRecord):
        """应用自适应增强"""
        # 记录参数历史
        self.param_history.append({
            "timestamp": time.time(),
            "params": asdict(self.adaptive_params),
            "strategy": self.current_strategy.value,
            "performance": evolution_record.metrics.success_rate
        })
        
        # 增强记忆整合
        if self.enhanced_memory and ENHANCED_MEMORY_AVAILABLE:
            self._integrate_enhanced_memory(evolution_record)
        
        # 动态调整学习率
        self._adjust_learning_rate(evolution_record)
    
    def _integrate_enhanced_memory(self, evolution_record: EvolutionRecord):
        """整合增强记忆"""
        if not self.enhanced_memory:
            return
        
        try:
            # 创建进化记忆
            evolution_memory = EnhancedMemory(
                content=f"自适应进化 {evolution_record.version}",
                memory_type="evolution",
                importance=0.9,
                emotional_valence=0.1 if evolution_record.metrics.success_rate > 0.7 else -0.1,
                tags=["adaptive_evolution", self.current_strategy.value],
                metadata={
                    "strategy": self.current_strategy.value,
                    "parameters": asdict(self.adaptive_params),
                    "metrics": asdict(evolution_record.metrics)
                }
            )
            
            self.enhanced_memory.add_memory(evolution_memory)
            
            # 执行记忆巩固
            self.enhanced_memory.consolidate_memories()
            
            # 应用遗忘曲线
            self.enhanced_memory.apply_forgetting_curve()
            
        except Exception as e:
            print(f"增强记忆整合失败: {e}")
    
    def _adjust_learning_rate(self, evolution_record: EvolutionRecord):
        """动态调整学习率"""
        current_performance = evolution_record.metrics.success_rate
        
        if len(self.performance_history) >= 2:
            previous_performance = self.performance_history[-2]
            improvement = current_performance - previous_performance
            
            if improvement > 0.1:
                # 显著改进 -> 降低学习率（稳定化）
                self.adaptive_params.learning_rate *= 0.95
            elif improvement < -0.1:
                # 性能下降 -> 增加学习率（加速适应）
                self.adaptive_params.learning_rate *= 1.05
            
            # 限制学习率范围
            self.adaptive_params.learning_rate = max(0.01, min(0.5, self.adaptive_params.learning_rate))
    
    def _update_strategy_performance(self, evolution_record: EvolutionRecord):
        """更新策略性能"""
        current_performance = evolution_record.metrics.success_rate
        
        # 计算改进程度
        if len(self.performance_history) >= 2:
            previous_performance = self.performance_history[-2]
            improvement = current_performance - previous_performance
        else:
            improvement = 0.0
        
        # 更新当前策略的性能
        self.strategy_performances[self.current_strategy].update_performance(improvement)
    
    def _update_pareto_archive(self, evolution_record: EvolutionRecord):
        """更新帕累托档案"""
        # 构建目标向量
        objectives = {
            OptimizationObjective.SUCCESS_RATE: evolution_record.metrics.success_rate,
            OptimizationObjective.RESPONSE_QUALITY: evolution_record.metrics.response_quality,
            OptimizationObjective.LEARNING_EFFICIENCY: evolution_record.metrics.learning_efficiency,
            OptimizationObjective.ADAPTATION_SPEED: evolution_record.metrics.adaptation_speed,
            OptimizationObjective.USER_SATISFACTION: evolution_record.metrics.user_satisfaction
        }
        
        # 只考虑当前选择的目标
        current_solution = {
            obj.value: objectives[obj] for obj in self.optimization_objectives
        }
        current_solution["timestamp"] = time.time()
        current_solution["strategy"] = self.current_strategy.value
        
        # 检查是否被现有解支配
        is_dominated = False
        dominated_indices = []
        
        for i, existing_solution in enumerate(self.pareto_archive):
            if self._dominates(existing_solution, current_solution):
                is_dominated = True
                break
            elif self._dominates(current_solution, existing_solution):
                dominated_indices.append(i)
        
        if not is_dominated:
            # 移除被新解支配的解
            for i in reversed(dominated_indices):
                self.pareto_archive.pop(i)
            
            # 添加新解
            self.pareto_archive.append(current_solution)
            
            # 限制档案大小
            if len(self.pareto_archive) > 50:
                self.pareto_archive = self._maintain_diversity(self.pareto_archive, 50)
    
    def _dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """检查solution1是否支配solution2"""
        better_in_any = False
        
        for obj in self.optimization_objectives:
            obj_name = obj.value
            if obj_name in solution1 and obj_name in solution2:
                if solution1[obj_name] < solution2[obj_name]:
                    return False
                elif solution1[obj_name] > solution2[obj_name]:
                    better_in_any = True
        
        return better_in_any
    
    def _maintain_diversity(self, solutions: List[Dict[str, float]], max_size: int) -> List[Dict[str, float]]:
        """维护解的多样性"""
        if len(solutions) <= max_size:
            return solutions
        
        # 简化的多样性维护：基于欧氏距离
        selected = [solutions[0]]  # 保留第一个解
        
        while len(selected) < max_size and len(selected) < len(solutions):
            max_min_distance = 0
            best_candidate = None
            
            for candidate in solutions:
                if candidate in selected:
                    continue
                
                min_distance = float('inf')
                for selected_solution in selected:
                    distance = self._calculate_solution_distance(candidate, selected_solution)
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_solution_distance(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> float:
        """计算解之间的距离"""
        distance = 0.0
        count = 0
        
        for obj in self.optimization_objectives:
            obj_name = obj.value
            if obj_name in solution1 and obj_name in solution2:
                distance += (solution1[obj_name] - solution2[obj_name]) ** 2
                count += 1
        
        return math.sqrt(distance / max(1, count))
    
    def get_multi_objective_analysis(self) -> MultiObjectiveResult:
        """获取多目标分析结果"""
        if not self.pareto_archive:
            return MultiObjectiveResult(
                pareto_front=[],
                hypervolume=0.0,
                diversity_metric=0.0,
                convergence_metric=0.0,
                dominated_solutions=0
            )
        
        # 计算超体积（简化版）
        reference_point = {obj.value: 0.0 for obj in self.optimization_objectives}
        hypervolume = self._calculate_hypervolume(self.pareto_archive, reference_point)
        
        # 计算多样性指标
        diversity_metric = self._calculate_diversity_metric(self.pareto_archive)
        
        # 计算收敛性指标（与理想点的距离）
        ideal_point = {obj.value: 1.0 for obj in self.optimization_objectives}
        convergence_metric = self._calculate_convergence_metric(self.pareto_archive, ideal_point)
        
        return MultiObjectiveResult(
            pareto_front=self.pareto_archive.copy(),
            hypervolume=hypervolume,
            diversity_metric=diversity_metric,
            convergence_metric=convergence_metric,
            dominated_solutions=0  # 帕累托前沿中没有被支配的解
        )
    
    def _calculate_hypervolume(self, solutions: List[Dict[str, float]], reference: Dict[str, float]) -> float:
        """计算超体积（简化版）"""
        if not solutions:
            return 0.0
        
        total_volume = 0.0
        
        for solution in solutions:
            volume = 1.0
            for obj in self.optimization_objectives:
                obj_name = obj.value
                if obj_name in solution and obj_name in reference:
                    volume *= max(0.0, solution[obj_name] - reference[obj_name])
            total_volume += volume
        
        return total_volume
    
    def _calculate_diversity_metric(self, solutions: List[Dict[str, float]]) -> float:
        """计算多样性指标"""
        if len(solutions) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                distance = self._calculate_solution_distance(solutions[i], solutions[j])
                total_distance += distance
                count += 1
        
        return total_distance / max(1, count)
    
    def _calculate_convergence_metric(self, solutions: List[Dict[str, float]], ideal: Dict[str, float]) -> float:
        """计算收敛性指标"""
        if not solutions:
            return float('inf')
        
        total_distance = 0.0
        
        for solution in solutions:
            distance = self._calculate_solution_distance(solution, ideal)
            total_distance += distance
        
        return total_distance / len(solutions)
    
    def get_adaptive_evolution_summary(self) -> Dict[str, Any]:
        """获取自适应进化摘要"""
        # 策略性能统计
        strategy_stats = {}
        for strategy, performance in self.strategy_performances.items():
            strategy_stats[strategy.value] = {
                "success_rate": performance.success_rate,
                "avg_improvement": performance.avg_improvement,
                "confidence": performance.confidence,
                "total_attempts": performance.total_attempts
            }
        
        # 多目标分析
        mo_analysis = self.get_multi_objective_analysis()
        
        # 环境适应统计
        adaptation_stats = {
            "total_adaptations": len(self.environment_changes),
            "recent_adaptations": len([
                change for change in self.environment_changes
                if time.time() - change["timestamp"] < 3600  # 最近1小时
            ])
        }
        
        return {
            "current_strategy": self.current_strategy.value,
            "adaptive_parameters": asdict(self.adaptive_params),
            "strategy_performance": strategy_stats,
            "multi_objective_analysis": {
                "pareto_front_size": len(mo_analysis.pareto_front),
                "hypervolume": mo_analysis.hypervolume,
                "diversity_metric": mo_analysis.diversity_metric,
                "convergence_metric": mo_analysis.convergence_metric
            },
            "environment_adaptation": adaptation_stats,
            "performance_trend": self._calculate_performance_trend(),
            "stagnation_counter": self.stagnation_counter,
            "parameter_evolution_history": len(self.param_history)
        }
    
    def _calculate_performance_trend(self) -> str:
        """计算性能趋势"""
        if len(self.performance_history) < 10:
            return "数据不足"
        
        recent_data = list(self.performance_history)
        if len(recent_data) >= 20:
            recent_avg = statistics.mean(recent_data[-10:])
            earlier_avg = statistics.mean(recent_data[-20:-10])
            
            improvement = recent_avg - earlier_avg
            
            if improvement > 0.15:
                return "显著改善"
            elif improvement > 0.05:
                return "轻微改善"
            elif improvement > -0.05:
                return "稳定"
            elif improvement > -0.15:
                return "轻微下降"
            else:
                return "显著下降"
        
        return "稳定"
    
    def save_adaptive_data(self):
        """保存自适应数据"""
        adaptive_data = {
            "strategy_performances": {
                strategy.value: {
                    "success_count": perf.success_count,
                    "total_attempts": perf.total_attempts,
                    "avg_improvement": perf.avg_improvement,
                    "recent_performance": list(perf.recent_performance),
                    "confidence": perf.confidence
                }
                for strategy, perf in self.strategy_performances.items()
            },
            "current_strategy": self.current_strategy.value,
            "adaptive_params": asdict(self.adaptive_params),
            "param_history": self.param_history,
            "pareto_archive": self.pareto_archive,
            "environment_changes": self.environment_changes,
            "performance_history": list(self.performance_history),
            "stagnation_counter": self.stagnation_counter,
            "last_significant_improvement": self.last_significant_improvement
        }
        
        with open("adaptive_evolution_data.json", "w", encoding="utf-8") as f:
            json.dump(adaptive_data, f, indent=2, ensure_ascii=False)
    
    def load_adaptive_data(self):
        """加载自适应数据"""
        try:
            with open("adaptive_evolution_data.json", "r", encoding="utf-8") as f:
                adaptive_data = json.load(f)
            
            # 加载策略性能
            for strategy_name, perf_data in adaptive_data.get("strategy_performances", {}).items():
                try:
                    strategy = EvolutionStrategy(strategy_name)
                    performance = StrategyPerformance(strategy)
                    performance.success_count = perf_data.get("success_count", 0)
                    performance.total_attempts = perf_data.get("total_attempts", 0)
                    performance.avg_improvement = perf_data.get("avg_improvement", 0.0)
                    performance.confidence = perf_data.get("confidence", 0.5)
                    performance.recent_performance = deque(
                        perf_data.get("recent_performance", []), maxlen=20
                    )
                    self.strategy_performances[strategy] = performance
                except ValueError:
                    continue
            
            # 加载当前策略
            try:
                self.current_strategy = EvolutionStrategy(
                    adaptive_data.get("current_strategy", "balanced")
                )
            except ValueError:
                self.current_strategy = EvolutionStrategy.BALANCED
            
            # 加载自适应参数
            if "adaptive_params" in adaptive_data:
                param_data = adaptive_data["adaptive_params"]
                self.adaptive_params = AdaptiveParameters(
                    learning_rate=param_data.get("learning_rate", 0.1),
                    mutation_rate=param_data.get("mutation_rate", 0.1),
                    selection_pressure=param_data.get("selection_pressure", 0.8),
                    diversity_threshold=param_data.get("diversity_threshold", 0.3),
                    convergence_threshold=param_data.get("convergence_threshold", 0.01),
                    exploration_rate=param_data.get("exploration_rate", 0.2),
                    temperature=param_data.get("temperature", 1.0)
                )
            
            # 加载其他数据
            self.param_history = adaptive_data.get("param_history", [])
            self.pareto_archive = adaptive_data.get("pareto_archive", [])
            self.environment_changes = adaptive_data.get("environment_changes", [])
            self.performance_history = deque(
                adaptive_data.get("performance_history", []), maxlen=100
            )
            self.stagnation_counter = adaptive_data.get("stagnation_counter", 0)
            self.last_significant_improvement = adaptive_data.get(
                "last_significant_improvement", time.time()
            )
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"加载自适应数据失败: {e}")