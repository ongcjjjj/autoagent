"""
统一进化系统 (Unified Evolution System)
整合并优化所有进化相关功能的核心模块

整合内容：
1. 原始进化引擎 (evolution.py)
2. 遗传进化算法 (genetic_evolution.py) 
3. 自适应进化引擎 (adaptive_evolution.py)
4. 自然启发进化 (natural_inspired_evolution.py)
5. 增强记忆系统 (enhanced_memory.py)
"""

import sqlite3
import json
import time
import random
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查numpy可用性
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    logger.warning("NumPy未安装，将使用简化数学运算")
    HAS_NUMPY = False

class EvolutionStrategy(Enum):
    """进化策略类型"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"  
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"

class ParticleType(Enum):
    """粒子类型"""
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"
    BALANCED = "balanced"

@dataclass
class EvolutionMetrics:
    """进化指标"""
    success_rate: float = 0.0
    response_quality: float = 0.0
    learning_efficiency: float = 0.0
    adaptation_speed: float = 0.0
    user_satisfaction: float = 0.0
    diversity_index: float = 0.0
    convergence_rate: float = 0.0
    stability_measure: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class EnhancedMemory:
    """增强记忆结构"""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "conversation"
    importance: float = 0.5
    emotional_valence: float = 0.0  # 情感价值 (-1到1)
    access_frequency: int = 0
    consolidation_level: int = 0  # 巩固级别 0-3
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    embedding_vector: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    associations: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvolutionParticle:
    """进化粒子"""
    id: str
    position: List[float]
    velocity: List[float]
    fitness: float = 0.0
    particle_type: ParticleType = ParticleType.BALANCED
    energy: float = 1.0
    temperature: float = 1.0
    age: int = 0
    generation: int = 0
    species: str = "default"
    traits: Dict[str, Any] = field(default_factory=dict)

class UnifiedEvolutionSystem:
    """统一进化系统"""
    
    def __init__(self, db_path: str = "unified_evolution.db"):
        self.db_path = db_path
        self.performance_window = deque(maxlen=1000)
        self.evolution_history = []
        self.adaptation_rules = {}
        self.learning_patterns = {}
        
        # 遗传算法参数
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_ratio = 0.2
        
        # 粒子群参数
        self.swarm_size = 30
        self.w_inertia = 0.7
        self.c1_cognitive = 1.4
        self.c2_social = 1.4
        
        # 自适应参数
        self.strategy_weights = {
            EvolutionStrategy.EXPLORATION: 0.25,
            EvolutionStrategy.EXPLOITATION: 0.25,
            EvolutionStrategy.BALANCED: 0.25,
            EvolutionStrategy.ADAPTIVE: 0.25
        }
        
        # 初始化组件
        self.init_database()
        self.particles = []
        self.population = []
        self.memory_graph = {}
        self.load_evolution_data()
        
        logger.info("统一进化系统初始化完成")
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 增强记忆表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'conversation',
                importance REAL DEFAULT 0.5,
                emotional_valence REAL DEFAULT 0.0,
                access_frequency INTEGER DEFAULT 0,
                consolidation_level INTEGER DEFAULT 0,
                creation_time REAL NOT NULL,
                last_access_time REAL NOT NULL,
                embedding_vector TEXT,
                tags TEXT DEFAULT '[]',
                associations TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # 记忆关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id_1 INTEGER,
                memory_id_2 INTEGER,
                association_strength REAL DEFAULT 0.0,
                association_type TEXT DEFAULT 'semantic',
                created_time REAL NOT NULL,
                FOREIGN KEY (memory_id_1) REFERENCES enhanced_memories (id),
                FOREIGN KEY (memory_id_2) REFERENCES enhanced_memories (id)
            )
        """)
        
        # 进化记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                strategy TEXT NOT NULL,
                metrics TEXT NOT NULL,
                improvements TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        # 粒子状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS particle_states (
                id TEXT PRIMARY KEY,
                position TEXT NOT NULL,
                velocity TEXT NOT NULL,
                fitness REAL DEFAULT 0.0,
                particle_type TEXT DEFAULT 'balanced',
                energy REAL DEFAULT 1.0,
                temperature REAL DEFAULT 1.0,
                age INTEGER DEFAULT 0,
                generation INTEGER DEFAULT 0,
                species TEXT DEFAULT 'default',
                traits TEXT DEFAULT '{}',
                timestamp REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    # === 记忆管理功能 ===
    
    def add_enhanced_memory(self, memory: EnhancedMemory) -> int:
        """添加增强记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO enhanced_memories 
            (content, memory_type, importance, emotional_valence, access_frequency,
             consolidation_level, creation_time, last_access_time, embedding_vector,
             tags, associations, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.content, memory.memory_type, memory.importance,
            memory.emotional_valence, memory.access_frequency,
            memory.consolidation_level, memory.creation_time,
            memory.last_access_time, json.dumps(memory.embedding_vector),
            json.dumps(memory.tags), json.dumps(memory.associations),
            json.dumps(memory.metadata)
        ))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 自动建立关联
        self.auto_associate_memory(memory_id)
        return memory_id or 0
    
    def search_enhanced_memories(self, query: str, limit: int = 10) -> List[EnhancedMemory]:
        """智能搜索增强记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基于内容的搜索
        cursor.execute("""
            SELECT * FROM enhanced_memories 
            WHERE content LIKE ?
            ORDER BY importance DESC, access_frequency DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        memories = [self._row_to_enhanced_memory(row) for row in rows]
        
        # 更新访问频次
        for memory in memories:
            self.update_memory_access(memory.id)
        
        return memories
    
    def consolidate_memories(self):
        """记忆巩固机制"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取需要巩固的记忆
        cursor.execute("""
            SELECT * FROM enhanced_memories 
            WHERE consolidation_level < 3 
            AND access_frequency > 0
            ORDER BY importance DESC, access_frequency DESC
        """)
        
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_enhanced_memory(row)
            
            # 计算巩固概率
            consolidation_prob = self._calculate_consolidation_probability(memory)
            
            if random.random() < consolidation_prob:
                new_level = min(memory.consolidation_level + 1, 3)
                cursor.execute("""
                    UPDATE enhanced_memories 
                    SET consolidation_level = ?, importance = importance * 1.1
                    WHERE id = ?
                """, (new_level, memory.id))
        
        conn.commit()
        conn.close()
    
    def apply_forgetting_curve(self):
        """应用遗忘曲线"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = time.time()
        
        cursor.execute("SELECT * FROM enhanced_memories")
        rows = cursor.fetchall()
        
        for row in rows:
            memory = self._row_to_enhanced_memory(row)
            
            # 计算遗忘衰减
            time_diff = current_time - memory.last_access_time
            days_passed = time_diff / (24 * 3600)
            
            # 考虑巩固级别的遗忘速度
            forgetting_rate = 0.1 / (memory.consolidation_level + 1)
            decay_factor = math.exp(-forgetting_rate * days_passed)
            
            new_importance = memory.importance * decay_factor
            
            cursor.execute("""
                UPDATE enhanced_memories 
                SET importance = ?
                WHERE id = ?
            """, (new_importance, memory.id))
        
        conn.commit()
        conn.close()
    
    # === 遗传算法功能 ===
    
    def initialize_population(self, size: int = None):
        """初始化种群"""
        if size is None:
            size = self.population_size
        
        self.population = []
        for i in range(size):
            individual = {
                'id': f"ind_{i}",
                'genes': [random.random() for _ in range(10)],  # 10维基因
                'fitness': 0.0,
                'age': 0,
                'generation': 0
            }
            self.population.append(individual)
    
    def evaluate_fitness(self, individual: Dict[str, Any]) -> float:
        """评估个体适应度"""
        genes = individual['genes']
        
        # 多目标适应度函数
        objectives = []
        
        # 目标1：性能指标
        performance = sum(genes[:3]) / 3
        objectives.append(performance)
        
        # 目标2：稳定性
        stability = 1.0 - statistics.variance(genes[3:6])
        objectives.append(max(0, stability))
        
        # 目标3：适应性
        adaptability = sum(abs(g - 0.5) for g in genes[6:]) / 4
        objectives.append(1.0 - adaptability)
        
        # 加权组合
        fitness = sum(obj * weight for obj, weight in zip(objectives, [0.4, 0.3, 0.3]))
        return max(0, fitness)
    
    def selection_tournament(self, tournament_size: int = 3) -> Dict[str, Any]:
        """锦标赛选择"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x['fitness'])
    
    def crossover_arithmetic(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """算术交叉"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = random.random()
        
        child1_genes = []
        child2_genes = []
        
        for g1, g2 in zip(parent1['genes'], parent2['genes']):
            child1_genes.append(alpha * g1 + (1 - alpha) * g2)
            child2_genes.append((1 - alpha) * g1 + alpha * g2)
        
        child1 = {
            'id': f"child_{len(self.population)}",
            'genes': child1_genes,
            'fitness': 0.0,
            'age': 0,
            'generation': parent1['generation'] + 1
        }
        
        child2 = {
            'id': f"child_{len(self.population) + 1}",
            'genes': child2_genes,
            'fitness': 0.0,
            'age': 0,
            'generation': parent1['generation'] + 1
        }
        
        return child1, child2
    
    def mutate_gaussian(self, individual: Dict[str, Any]):
        """高斯变异"""
        for i in range(len(individual['genes'])):
            if random.random() < self.mutation_rate:
                individual['genes'][i] += random.gauss(0, 0.1)
                individual['genes'][i] = max(0, min(1, individual['genes'][i]))
    
    def evolve_generation(self) -> Dict[str, Any]:
        """进化一代"""
        if not self.population:
            self.initialize_population()
        
        # 评估适应度
        for individual in self.population:
            individual['fitness'] = self.evaluate_fitness(individual)
        
        # 选择精英
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        elite_count = int(len(self.population) * self.elite_ratio)
        elite = self.population[:elite_count]
        
        # 生成新一代
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            parent1 = self.selection_tournament()
            parent2 = self.selection_tournament()
            
            child1, child2 = self.crossover_arithmetic(parent1, parent2)
            
            self.mutate_gaussian(child1)
            self.mutate_gaussian(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        
        # 返回进化统计
        fitnesses = [ind['fitness'] for ind in self.population]
        return {
            'generation': self.population[0]['generation'],
            'best_fitness': max(fitnesses),
            'average_fitness': statistics.mean(fitnesses),
            'diversity': statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0
        }
    
    # === 粒子群算法功能 ===
    
    def initialize_swarm(self, dimensions: int = 10):
        """初始化粒子群"""
        self.particles = []
        for i in range(self.swarm_size):
            particle = EvolutionParticle(
                id=f"particle_{i}",
                position=[random.random() for _ in range(dimensions)],
                velocity=[random.uniform(-0.1, 0.1) for _ in range(dimensions)],
                particle_type=random.choice(list(ParticleType))
            )
            self.particles.append(particle)
    
    def update_particle_swarm(self) -> Dict[str, Any]:
        """更新粒子群"""
        if not self.particles:
            self.initialize_swarm()
        
        # 评估适应度
        for particle in self.particles:
            particle.fitness = self._evaluate_particle_fitness(particle)
        
        # 找到全局最优
        global_best = max(self.particles, key=lambda p: p.fitness)
        
        # 更新粒子
        for particle in self.particles:
            # 个体最优（简化版）
            personal_best = particle.position.copy()
            
            # 更新速度和位置
            for i in range(len(particle.position)):
                r1, r2 = random.random(), random.random()
                
                cognitive = self.c1_cognitive * r1 * (personal_best[i] - particle.position[i])
                social = self.c2_social * r2 * (global_best.position[i] - particle.position[i])
                
                particle.velocity[i] = (self.w_inertia * particle.velocity[i] + 
                                       cognitive + social)
                
                # 限制速度
                particle.velocity[i] = max(-0.5, min(0.5, particle.velocity[i]))
                
                # 更新位置
                particle.position[i] += particle.velocity[i]
                particle.position[i] = max(0, min(1, particle.position[i]))
            
            # 更新粒子属性
            particle.age += 1
            particle.energy *= 0.99  # 能量衰减
        
        # 返回群体统计
        fitnesses = [p.fitness for p in self.particles]
        return {
            'best_fitness': max(fitnesses),
            'average_fitness': statistics.mean(fitnesses),
            'swarm_diversity': self._calculate_swarm_diversity()
        }
    
    # === 自适应策略功能 ===
    
    def select_evolution_strategy(self) -> EvolutionStrategy:
        """智能选择进化策略"""
        if len(self.performance_window) < 20:
            return EvolutionStrategy.BALANCED
        
        recent_performance = list(self.performance_window)[-20:]
        performance_trend = self._calculate_performance_trend(recent_performance)
        diversity = self._calculate_population_diversity()
        
        # ε-贪心策略选择
        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(list(EvolutionStrategy))
        
        # 基于性能选择策略
        if performance_trend < -0.1:  # 性能下降
            if diversity < 0.3:  # 多样性不足
                return EvolutionStrategy.EXPLORATION
            else:
                return EvolutionStrategy.ADAPTIVE
        elif performance_trend > 0.1:  # 性能上升
            return EvolutionStrategy.EXPLOITATION
        else:  # 稳定期
            return EvolutionStrategy.BALANCED
    
    def execute_unified_evolution(self) -> Dict[str, Any]:
        """执行统一进化"""
        start_time = time.time()
        
        # 选择进化策略
        strategy = self.select_evolution_strategy()
        
        results = {}
        
        # 根据策略执行不同的进化操作
        if strategy == EvolutionStrategy.EXPLORATION:
            # 增加变异率，增强探索
            old_mutation_rate = self.mutation_rate
            self.mutation_rate *= 1.5
            results['genetic'] = self.evolve_generation()
            self.mutation_rate = old_mutation_rate
            
        elif strategy == EvolutionStrategy.EXPLOITATION:
            # 减少变异率，加强开发
            old_mutation_rate = self.mutation_rate
            self.mutation_rate *= 0.5
            results['genetic'] = self.evolve_generation()
            self.mutation_rate = old_mutation_rate
            
        elif strategy == EvolutionStrategy.BALANCED:
            # 平衡策略
            results['genetic'] = self.evolve_generation()
            results['swarm'] = self.update_particle_swarm()
            
        else:  # ADAPTIVE
            # 自适应策略，同时运行多种算法
            results['genetic'] = self.evolve_generation()
            results['swarm'] = self.update_particle_swarm()
            
            # 记忆巩固
            self.consolidate_memories()
            self.apply_forgetting_curve()
        
        # 更新策略权重
        self._update_strategy_weights(strategy, results)
        
        # 记录进化事件
        evolution_record = {
            'strategy': strategy.value,
            'results': results,
            'execution_time': time.time() - start_time,
            'timestamp': time.time()
        }
        
        self.evolution_history.append(evolution_record)
        self.save_evolution_data()
        
        return evolution_record
    
    # === 辅助方法 ===
    
    def _row_to_enhanced_memory(self, row) -> EnhancedMemory:
        """数据库行转增强记忆对象"""
        return EnhancedMemory(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            importance=row[3],
            emotional_valence=row[4],
            access_frequency=row[5],
            consolidation_level=row[6],
            creation_time=row[7],
            last_access_time=row[8],
            embedding_vector=json.loads(row[9]) if row[9] else None,
            tags=json.loads(row[10]),
            associations=json.loads(row[11]),
            metadata=json.loads(row[12])
        )
    
    def _calculate_consolidation_probability(self, memory: EnhancedMemory) -> float:
        """计算记忆巩固概率"""
        importance_factor = memory.importance
        access_factor = min(memory.access_frequency / 10.0, 1.0)
        time_factor = min((time.time() - memory.creation_time) / (7 * 24 * 3600), 1.0)
        
        return importance_factor * 0.5 + access_factor * 0.3 + time_factor * 0.2
    
    def _evaluate_particle_fitness(self, particle: EvolutionParticle) -> float:
        """评估粒子适应度"""
        position_sum = sum(particle.position)
        position_variance = statistics.variance(particle.position) if len(particle.position) > 1 else 0
        
        fitness = position_sum / len(particle.position) - position_variance * 0.1
        return max(0, fitness)
    
    def _calculate_swarm_diversity(self) -> float:
        """计算群体多样性"""
        if len(self.particles) < 2:
            return 0.0
        
        distances = []
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                dist = sum((a - b) ** 2 for a, b in 
                          zip(self.particles[i].position, self.particles[j].position)) ** 0.5
                distances.append(dist)
        
        return statistics.mean(distances) if distances else 0.0
    
    def _calculate_performance_trend(self, performance_data: List[Dict]) -> float:
        """计算性能趋势"""
        if len(performance_data) < 10:
            return 0.0
        
        scores = [item.get('score', 0) for item in performance_data]
        mid_point = len(scores) // 2
        
        early_avg = statistics.mean(scores[:mid_point])
        recent_avg = statistics.mean(scores[mid_point:])
        
        return recent_avg - early_avg
    
    def _calculate_population_diversity(self) -> float:
        """计算种群多样性"""
        if not self.population or len(self.population) < 2:
            return 0.0
        
        all_genes = [ind['genes'] for ind in self.population]
        diversities = []
        
        for i in range(len(all_genes[0])):
            gene_values = [genes[i] for genes in all_genes]
            diversity = statistics.stdev(gene_values) if len(gene_values) > 1 else 0
            diversities.append(diversity)
        
        return statistics.mean(diversities)
    
    def _update_strategy_weights(self, strategy: EvolutionStrategy, results: Dict[str, Any]):
        """更新策略权重"""
        # 简化的权重更新机制
        performance = 0.0
        count = 0
        
        for result in results.values():
            if isinstance(result, dict) and 'best_fitness' in result:
                performance += result['best_fitness']
                count += 1
        
        if count > 0:
            avg_performance = performance / count
            
            # 更新权重
            current_weight = self.strategy_weights[strategy]
            learning_rate = 0.1
            
            new_weight = current_weight + learning_rate * (avg_performance - 0.5)
            self.strategy_weights[strategy] = max(0.1, min(0.9, new_weight))
            
            # 归一化权重
            total_weight = sum(self.strategy_weights.values())
            for s in self.strategy_weights:
                self.strategy_weights[s] /= total_weight
    
    def auto_associate_memory(self, memory_id: int):
        """自动建立记忆关联"""
        # 简化版本的自动关联
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT content FROM enhanced_memories WHERE id = ?", (memory_id,))
        target_memory = cursor.fetchone()
        
        if target_memory:
            target_content = target_memory[0]
            
            # 找到相似记忆
            cursor.execute("""
                SELECT id, content FROM enhanced_memories 
                WHERE id != ? 
                ORDER BY creation_time DESC 
                LIMIT 20
            """, (memory_id,))
            
            similar_memories = cursor.fetchall()
            
            for mem_id, content in similar_memories:
                # 简单的文本相似度计算
                similarity = self._calculate_text_similarity(target_content, content)
                
                if similarity > 0.3:  # 阈值
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_associations 
                        (memory_id_1, memory_id_2, association_strength, 
                         association_type, created_time)
                        VALUES (?, ?, ?, ?, ?)
                    """, (memory_id, mem_id, similarity, 'semantic', time.time()))
        
        conn.commit()
        conn.close()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def update_memory_access(self, memory_id: int):
        """更新记忆访问信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE enhanced_memories 
            SET access_frequency = access_frequency + 1,
                last_access_time = ?
            WHERE id = ?
        """, (time.time(), memory_id))
        
        conn.commit()
        conn.close()
    
    def save_evolution_data(self):
        """保存进化数据"""
        # 转换策略权重为可序列化格式
        serializable_weights = {strategy.value: weight for strategy, weight in self.strategy_weights.items()}
        
        data = {
            'strategy_weights': serializable_weights,
            'adaptation_rules': self.adaptation_rules,
            'learning_patterns': self.learning_patterns,
            'evolution_history': self.evolution_history[-100:],  # 只保存最近100条
            'timestamp': time.time()
        }
        
        with open('unified_evolution_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_evolution_data(self):
        """加载进化数据"""
        try:
            with open('unified_evolution_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换字符串格式的策略权重回枚举类型
            strategy_weights_data = data.get('strategy_weights', {})
            for strategy_str, weight in strategy_weights_data.items():
                for strategy in EvolutionStrategy:
                    if strategy.value == strategy_str:
                        self.strategy_weights[strategy] = weight
                        break
            
            self.adaptation_rules.update(data.get('adaptation_rules', {}))
            self.learning_patterns.update(data.get('learning_patterns', {}))
            self.evolution_history = data.get('evolution_history', [])
            
        except FileNotFoundError:
            logger.info("未找到进化数据文件，使用默认设置")
        except Exception as e:
            logger.error(f"加载进化数据失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 转换策略权重为可序列化格式
        serializable_weights = {strategy.value: weight for strategy, weight in self.strategy_weights.items()}
        
        return {
            'population_size': len(self.population),
            'swarm_size': len(self.particles),
            'strategy_weights': serializable_weights,
            'evolution_count': len(self.evolution_history),
            'performance_window_size': len(self.performance_window),
            'database_path': self.db_path,
            'has_numpy': HAS_NUMPY
        }