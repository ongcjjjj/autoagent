"""
Agent进化机制模块
实现自我学习和进化功能
"""
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from memory import Memory, MemoryManager

@dataclass
class EvolutionMetrics:
    """进化指标"""
    success_rate: float = 0.0
    response_quality: float = 0.0
    learning_efficiency: float = 0.0
    adaptation_speed: float = 0.0
    user_satisfaction: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class EvolutionRecord:
    """进化记录"""
    version: str
    metrics: EvolutionMetrics
    changes: List[str]
    improvement_areas: List[str]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class EvolutionEngine:
    """进化引擎"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.evolution_history: List[EvolutionRecord] = []
        self.current_metrics = EvolutionMetrics()
        self.performance_window = []  # 性能窗口，存储最近的表现
        self.learning_patterns = {}   # 学习模式
        self.adaptation_rules = {}    # 适应规则
        
        self.load_evolution_data()
    
    def evaluate_performance(self, interaction_data: Dict[str, Any]) -> float:
        """
        评估单次交互的表现
        
        Args:
            interaction_data: 交互数据
            
        Returns:
            表现评分 (0-1)
        """
        score = 0.0
        factors = 0
        
        # 响应时间评分
        if 'response_time' in interaction_data:
            response_time = interaction_data['response_time']
            if response_time < 2.0:
                score += 1.0
            elif response_time < 5.0:
                score += 0.8
            elif response_time < 10.0:
                score += 0.6
            else:
                score += 0.3
            factors += 1
        
        # 用户反馈评分
        if 'user_feedback' in interaction_data:
            feedback = interaction_data['user_feedback']
            if feedback == 'excellent':
                score += 1.0
            elif feedback == 'good':
                score += 0.8
            elif feedback == 'average':
                score += 0.6
            elif feedback == 'poor':
                score += 0.3
            else:
                score += 0.5
            factors += 1
        
        # 任务完成率
        if 'task_completed' in interaction_data:
            score += 1.0 if interaction_data['task_completed'] else 0.0
            factors += 1
        
        # 错误率
        if 'error_count' in interaction_data:
            error_count = interaction_data['error_count']
            if error_count == 0:
                score += 1.0
            elif error_count <= 2:
                score += 0.7
            else:
                score += 0.3
            factors += 1
        
        return score / max(factors, 1)
    
    def update_performance_window(self, performance_score: float):
        """更新性能窗口"""
        self.performance_window.append({
            'score': performance_score,
            'timestamp': time.time()
        })
        
        # 保持窗口大小
        max_window_size = 100
        if len(self.performance_window) > max_window_size:
            self.performance_window = self.performance_window[-max_window_size:]
    
    def calculate_evolution_metrics(self) -> EvolutionMetrics:
        """计算进化指标"""
        if not self.performance_window:
            return EvolutionMetrics()
        
        scores = [item['score'] for item in self.performance_window]
        recent_scores = scores[-20:] if len(scores) >= 20 else scores
        
        # 成功率
        success_rate = statistics.mean(scores)
        
        # 响应质量（最近表现）
        response_quality = statistics.mean(recent_scores)
        
        # 学习效率（改进趋势）
        learning_efficiency = 0.0
        if len(scores) >= 10:
            early_avg = statistics.mean(scores[:len(scores)//2])
            recent_avg = statistics.mean(scores[len(scores)//2:])
            learning_efficiency = max(0.0, (recent_avg - early_avg) + 0.5)
        
        # 适应速度（方差的倒数）
        adaptation_speed = 0.0
        if len(recent_scores) > 1:
            variance = statistics.variance(recent_scores)
            adaptation_speed = 1.0 / (1.0 + variance)
        
        # 用户满意度（基于反馈）
        user_satisfaction = response_quality  # 简化版本
        
        return EvolutionMetrics(
            success_rate=success_rate,
            response_quality=response_quality,
            learning_efficiency=learning_efficiency,
            adaptation_speed=adaptation_speed,
            user_satisfaction=user_satisfaction
        )
    
    def identify_improvement_areas(self, metrics: EvolutionMetrics) -> List[str]:
        """识别改进领域"""
        improvement_areas = []
        
        if metrics.success_rate < 0.7:
            improvement_areas.append("提高任务成功率")
        
        if metrics.response_quality < 0.8:
            improvement_areas.append("改善响应质量")
        
        if metrics.learning_efficiency < 0.6:
            improvement_areas.append("增强学习效率")
        
        if metrics.adaptation_speed < 0.7:
            improvement_areas.append("提升适应能力")
        
        if metrics.user_satisfaction < 0.8:
            improvement_areas.append("提高用户满意度")
        
        return improvement_areas
    
    def generate_evolution_strategies(self, improvement_areas: List[str]) -> List[str]:
        """生成进化策略"""
        strategies = []
        
        strategy_map = {
            "提高任务成功率": [
                "优化任务分解逻辑",
                "增强错误处理机制",
                "改进任务理解能力"
            ],
            "改善响应质量": [
                "提升语言表达能力",
                "增强上下文理解",
                "优化回答结构"
            ],
            "增强学习效率": [
                "改进记忆存储策略",
                "优化知识提取方法",
                "加强模式识别能力"
            ],
            "提升适应能力": [
                "增强环境感知",
                "提高参数调整灵活性",
                "改善反馈响应机制"
            ],
            "提高用户满意度": [
                "个性化交互风格",
                "提升响应速度",
                "增强情感理解能力"
            ]
        }
        
        for area in improvement_areas:
            if area in strategy_map:
                strategies.extend(strategy_map[area])
        
        return list(set(strategies))  # 去重
    
    def should_evolve(self) -> bool:
        """判断是否应该进化"""
        if len(self.performance_window) < 50:
            return False
        
        current_metrics = self.calculate_evolution_metrics()
        
        # 如果表现持续低下，触发进化
        if current_metrics.success_rate < 0.6:
            return True
        
        # 如果学习效率低，触发进化
        if current_metrics.learning_efficiency < 0.3:
            return True
        
        # 如果用户满意度低，触发进化
        if current_metrics.user_satisfaction < 0.5:
            return True
        
        return False
    
    def execute_evolution(self) -> EvolutionRecord:
        """执行进化"""
        current_metrics = self.calculate_evolution_metrics()
        improvement_areas = self.identify_improvement_areas(current_metrics)
        evolution_strategies = self.generate_evolution_strategies(improvement_areas)
        
        # 创建进化记录
        evolution_record = EvolutionRecord(
            version=f"v{len(self.evolution_history) + 1}.0",
            metrics=current_metrics,
            changes=evolution_strategies,
            improvement_areas=improvement_areas
        )
        
        self.evolution_history.append(evolution_record)
        
        # 记录进化事件到记忆中
        evolution_memory = Memory(
            content=f"执行进化 {evolution_record.version}: {', '.join(improvement_areas)}",
            memory_type="evolution",
            importance=0.9,
            tags=["evolution", "improvement"],
            metadata={
                "version": evolution_record.version,
                "metrics": asdict(current_metrics),
                "strategies": evolution_strategies
            }
        )
        
        self.memory_manager.add_memory(evolution_memory)
        
        # 应用进化策略
        self.apply_evolution_strategies(evolution_strategies)
        
        # 保存进化数据
        self.save_evolution_data()
        
        return evolution_record
    
    def apply_evolution_strategies(self, strategies: List[str]):
        """应用进化策略"""
        for strategy in strategies:
            if "优化任务分解逻辑" in strategy:
                self.adaptation_rules['task_decomposition'] = 'enhanced'
            
            if "增强错误处理机制" in strategy:
                self.adaptation_rules['error_handling'] = 'robust'
            
            if "提升语言表达能力" in strategy:
                self.adaptation_rules['language_quality'] = 'improved'
            
            if "改进记忆存储策略" in strategy:
                self.adaptation_rules['memory_strategy'] = 'optimized'
            
            if "增强环境感知" in strategy:
                self.adaptation_rules['context_awareness'] = 'enhanced'
    
    def get_adaptation_rules(self) -> Dict[str, str]:
        """获取当前适应规则"""
        return self.adaptation_rules.copy()
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化摘要"""
        if not self.evolution_history:
            return {"message": "尚未进行进化"}
        
        latest_evolution = self.evolution_history[-1]
        current_metrics = self.calculate_evolution_metrics()
        
        return {
            "total_evolutions": len(self.evolution_history),
            "latest_version": latest_evolution.version,
            "current_metrics": asdict(current_metrics),
            "recent_improvements": latest_evolution.improvement_areas,
            "active_strategies": latest_evolution.changes,
            "performance_trend": self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """计算性能趋势"""
        if len(self.performance_window) < 20:
            return "数据不足"
        
        recent_scores = [item['score'] for item in self.performance_window[-20:]]
        early_scores = [item['score'] for item in self.performance_window[-40:-20]]
        
        if not early_scores:
            return "稳定"
        
        recent_avg = statistics.mean(recent_scores)
        early_avg = statistics.mean(early_scores)
        
        improvement = recent_avg - early_avg
        
        if improvement > 0.1:
            return "显著改善"
        elif improvement > 0.05:
            return "轻微改善"
        elif improvement > -0.05:
            return "稳定"
        elif improvement > -0.1:
            return "轻微下降"
        else:
            return "显著下降"
    
    def save_evolution_data(self):
        """保存进化数据"""
        evolution_data = {
            "evolution_history": [asdict(record) for record in self.evolution_history],
            "current_metrics": asdict(self.current_metrics),
            "adaptation_rules": self.adaptation_rules,
            "performance_window": self.performance_window[-100:]  # 只保存最近100条
        }
        
        with open("evolution_data.json", "w", encoding="utf-8") as f:
            json.dump(evolution_data, f, indent=2, ensure_ascii=False)
    
    def load_evolution_data(self):
        """加载进化数据"""
        try:
            with open("evolution_data.json", "r", encoding="utf-8") as f:
                evolution_data = json.load(f)
            
            # 加载进化历史
            self.evolution_history = [
                EvolutionRecord(
                    version=record["version"],
                    metrics=EvolutionMetrics(**record["metrics"]),
                    changes=record["changes"],
                    improvement_areas=record["improvement_areas"],
                    timestamp=record["timestamp"]
                )
                for record in evolution_data.get("evolution_history", [])
            ]
            
            # 加载当前指标
            if "current_metrics" in evolution_data:
                self.current_metrics = EvolutionMetrics(**evolution_data["current_metrics"])
            
            # 加载适应规则
            self.adaptation_rules = evolution_data.get("adaptation_rules", {})
            
            # 加载性能窗口
            self.performance_window = evolution_data.get("performance_window", [])
            
        except FileNotFoundError:
            # 如果文件不存在，使用默认值
            pass
        except Exception as e:
            print(f"加载进化数据失败: {e}")