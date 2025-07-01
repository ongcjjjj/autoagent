"""
遗传进化算法模块
实现差分进化、模因算法、协同进化等核心算法
"""
import numpy as np
import random
import copy
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import time

@dataclass
class Individual:
    """个体基类"""
    genes: List[float]  # 基因（参数向量）
    fitness: float = 0.0
    age: int = 0
    strategy: str = "default"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Population:
    """种群"""
    individuals: List[Individual]
    generation: int = 0
    diversity: float = 0.0
    avg_fitness: float = 0.0
    best_fitness: float = 0.0

class EvolutionAlgorithm(ABC):
    """进化算法基类"""
    
    @abstractmethod
    def initialize_population(self, size: int, dimension: int) -> Population:
        """初始化种群"""
        pass
    
    @abstractmethod
    def evolve(self, population: Population, fitness_func: Callable) -> Population:
        """进化一代"""
        pass

class DifferentialEvolution(EvolutionAlgorithm):
    """差分进化算法"""
    
    def __init__(self, F: float = 0.8, CR: float = 0.9, bounds: Tuple[float, float] = (-1.0, 1.0)):
        """
        初始化差分进化算法
        
        Args:
            F: 差分权重
            CR: 交叉概率
            bounds: 参数边界
        """
        self.F = F
        self.CR = CR
        self.bounds = bounds
        self.history = []
    
    def initialize_population(self, size: int, dimension: int) -> Population:
        """初始化种群"""
        individuals = []
        
        for i in range(size):
            genes = [random.uniform(self.bounds[0], self.bounds[1]) for _ in range(dimension)]
            individual = Individual(genes=genes, strategy="DE")
            individuals.append(individual)
        
        return Population(individuals=individuals)
    
    def evolve(self, population: Population, fitness_func: Callable) -> Population:
        """差分进化一代"""
        new_individuals = []
        
        for i, target in enumerate(population.individuals):
            # 选择三个不同的个体
            candidates = [j for j in range(len(population.individuals)) if j != i]
            if len(candidates) < 3:
                new_individuals.append(target)
                continue
            
            a, b, c = random.sample(candidates, 3)
            
            # 变异操作
            mutant = self._mutate(
                population.individuals[a],
                population.individuals[b],
                population.individuals[c]
            )
            
            # 交叉操作
            trial = self._crossover(target, mutant)
            
            # 选择操作
            trial.fitness = fitness_func(trial.genes)
            if target.fitness == 0.0:
                target.fitness = fitness_func(target.genes)
            
            if trial.fitness >= target.fitness:
                trial.age = target.age + 1
                new_individuals.append(trial)
            else:
                target.age += 1
                new_individuals.append(target)
        
        new_population = Population(
            individuals=new_individuals,
            generation=population.generation + 1
        )
        
        self._update_population_stats(new_population)
        return new_population
    
    def _mutate(self, a: Individual, b: Individual, c: Individual) -> Individual:
        """变异操作：mutant = a + F * (b - c)"""
        mutant_genes = []
        
        for i in range(len(a.genes)):
            gene = a.genes[i] + self.F * (b.genes[i] - c.genes[i])
            # 边界处理
            gene = max(self.bounds[0], min(self.bounds[1], gene))
            mutant_genes.append(gene)
        
        return Individual(genes=mutant_genes, strategy="DE_mutant")
    
    def _crossover(self, target: Individual, mutant: Individual) -> Individual:
        """交叉操作"""
        trial_genes = []
        
        # 确保至少有一个基因来自变异个体
        j_rand = random.randint(0, len(target.genes) - 1)
        
        for j in range(len(target.genes)):
            if random.random() < self.CR or j == j_rand:
                trial_genes.append(mutant.genes[j])
            else:
                trial_genes.append(target.genes[j])
        
        return Individual(genes=trial_genes, strategy="DE_trial")
    
    def _update_population_stats(self, population: Population):
        """更新种群统计信息"""
        if not population.individuals:
            return
        
        fitnesses = [ind.fitness for ind in population.individuals]
        population.avg_fitness = np.mean(fitnesses)
        population.best_fitness = max(fitnesses)
        
        # 计算多样性（基因的标准差）
        if len(population.individuals) > 1:
            gene_matrix = np.array([ind.genes for ind in population.individuals])
            population.diversity = np.mean(np.std(gene_matrix, axis=0))

class MemeticAlgorithm(EvolutionAlgorithm):
    """模因算法（遗传算法 + 局部搜索）"""
    
    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.8, 
                 local_search_rate: float = 0.3, bounds: Tuple[float, float] = (-1.0, 1.0)):
        """
        初始化模因算法
        
        Args:
            mutation_rate: 变异率
            crossover_rate: 交叉率
            local_search_rate: 局部搜索率
            bounds: 参数边界
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.local_search_rate = local_search_rate
        self.bounds = bounds
        self.elite_size = 5
    
    def initialize_population(self, size: int, dimension: int) -> Population:
        """初始化种群"""
        individuals = []
        
        for i in range(size):
            genes = [random.uniform(self.bounds[0], self.bounds[1]) for _ in range(dimension)]
            individual = Individual(genes=genes, strategy="MA")
            individuals.append(individual)
        
        return Population(individuals=individuals)
    
    def evolve(self, population: Population, fitness_func: Callable) -> Population:
        """模因算法进化一代"""
        # 计算适应度
        for individual in population.individuals:
            if individual.fitness == 0.0:
                individual.fitness = fitness_func(individual.genes)
        
        # 选择
        selected = self._tournament_selection(population.individuals, len(population.individuals))
        
        # 交叉和变异
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # 变异
            if random.random() < self.mutation_rate:
                self._mutate(child1)
            if random.random() < self.mutation_rate:
                self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        # 局部搜索
        for individual in offspring:
            if random.random() < self.local_search_rate:
                self._local_search(individual, fitness_func)
        
        # 环境选择（精英策略）
        all_individuals = population.individuals + offspring
        for individual in all_individuals:
            individual.fitness = fitness_func(individual.genes)
        
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)
        new_individuals = all_individuals[:len(population.individuals)]
        
        new_population = Population(
            individuals=new_individuals,
            generation=population.generation + 1
        )
        
        self._update_population_stats(new_population)
        return new_population
    
    def _tournament_selection(self, individuals: List[Individual], num_select: int, 
                            tournament_size: int = 3) -> List[Individual]:
        """锦标赛选择"""
        selected = []
        
        for _ in range(num_select):
            tournament = random.sample(individuals, min(tournament_size, len(individuals)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))
        
        return selected
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """算术交叉"""
        alpha = random.random()
        
        child1_genes = []
        child2_genes = []
        
        for i in range(len(parent1.genes)):
            gene1 = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i]
            gene2 = (1 - alpha) * parent1.genes[i] + alpha * parent2.genes[i]
            
            child1_genes.append(gene1)
            child2_genes.append(gene2)
        
        child1 = Individual(genes=child1_genes, strategy="MA_child")
        child2 = Individual(genes=child2_genes, strategy="MA_child")
        
        return child1, child2
    
    def _mutate(self, individual: Individual):
        """高斯变异"""
        for i in range(len(individual.genes)):
            if random.random() < 0.1:  # 每个基因10%的变异概率
                noise = random.gauss(0, 0.1)
                individual.genes[i] += noise
                # 边界处理
                individual.genes[i] = max(self.bounds[0], min(self.bounds[1], individual.genes[i]))
    
    def _local_search(self, individual: Individual, fitness_func: Callable):
        """简单的爬山局部搜索"""
        current_fitness = fitness_func(individual.genes)
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations:
            improved = False
            best_neighbor = copy.deepcopy(individual)
            best_fitness = current_fitness
            
            # 在每个维度上尝试小的改变
            for i in range(len(individual.genes)):
                for delta in [-0.05, 0.05]:
                    neighbor = copy.deepcopy(individual)
                    neighbor.genes[i] += delta
                    # 边界处理
                    neighbor.genes[i] = max(self.bounds[0], min(self.bounds[1], neighbor.genes[i]))
                    
                    neighbor_fitness = fitness_func(neighbor.genes)
                    if neighbor_fitness > best_fitness:
                        best_neighbor = neighbor
                        best_fitness = neighbor_fitness
                        improved = True
            
            if improved:
                individual.genes = best_neighbor.genes
                current_fitness = best_fitness
                iterations += 1
    
    def _update_population_stats(self, population: Population):
        """更新种群统计信息"""
        if not population.individuals:
            return
        
        fitnesses = [ind.fitness for ind in population.individuals]
        population.avg_fitness = np.mean(fitnesses)
        population.best_fitness = max(fitnesses)
        
        # 计算多样性
        if len(population.individuals) > 1:
            gene_matrix = np.array([ind.genes for ind in population.individuals])
            population.diversity = np.mean(np.std(gene_matrix, axis=0))

class CoevolutionaryAlgorithm:
    """协同进化算法"""
    
    def __init__(self, num_species: int = 3, species_size: int = 20):
        """
        初始化协同进化算法
        
        Args:
            num_species: 物种数量
            species_size: 每个物种的大小
        """
        self.num_species = num_species
        self.species_size = species_size
        self.species_populations = []
        self.interaction_history = []
    
    def initialize_species(self, dimension: int) -> List[Population]:
        """初始化所有物种"""
        self.species_populations = []
        
        for i in range(self.num_species):
            de = DifferentialEvolution()
            population = de.initialize_population(self.species_size, dimension)
            
            # 为每个物种分配特定的策略
            strategy_names = ["explorer", "exploiter", "balanced"]
            strategy = strategy_names[i % len(strategy_names)]
            
            for individual in population.individuals:
                individual.strategy = strategy
                individual.metadata["species_id"] = i
            
            self.species_populations.append(population)
        
        return self.species_populations
    
    def coevolve(self, fitness_func: Callable, generations: int = 100):
        """协同进化过程"""
        for generation in range(generations):
            # 为每个物种进化
            for species_id, population in enumerate(self.species_populations):
                # 创建特定的适应度函数，考虑与其他物种的交互
                species_fitness_func = self._create_species_fitness_func(
                    species_id, fitness_func
                )
                
                # 选择进化算法
                if population.individuals[0].strategy == "explorer":
                    algorithm = DifferentialEvolution(F=0.9, CR=0.7)
                elif population.individuals[0].strategy == "exploiter":
                    algorithm = MemeticAlgorithm(local_search_rate=0.5)
                else:
                    algorithm = DifferentialEvolution(F=0.8, CR=0.9)
                
                # 进化
                self.species_populations[species_id] = algorithm.evolve(
                    population, species_fitness_func
                )
            
            # 记录交互历史
            self._record_interaction(generation)
            
            # 物种间迁移
            if generation % 10 == 0:
                self._species_migration()
    
    def _create_species_fitness_func(self, species_id: int, base_fitness_func: Callable) -> Callable:
        """为特定物种创建适应度函数"""
        def species_fitness(genes: List[float]) -> float:
            # 基础适应度
            base_fitness = base_fitness_func(genes)
            
            # 与其他物种的协作适应度
            collaboration_bonus = 0.0
            
            for other_species_id, other_population in enumerate(self.species_populations):
                if other_species_id != species_id and other_population.individuals:
                    # 找到其他物种的最佳个体
                    best_other = max(other_population.individuals, key=lambda x: x.fitness)
                    
                    # 计算协作奖励
                    collaboration_bonus += self._calculate_collaboration(genes, best_other.genes)
            
            return base_fitness + 0.1 * collaboration_bonus / max(1, self.num_species - 1)
        
        return species_fitness
    
    def _calculate_collaboration(self, genes1: List[float], genes2: List[float]) -> float:
        """计算两个个体的协作程度"""
        if len(genes1) != len(genes2):
            return 0.0
        
        # 使用余弦相似度作为协作指标
        dot_product = sum(g1 * g2 for g1, g2 in zip(genes1, genes2))
        norm1 = math.sqrt(sum(g * g for g in genes1))
        norm2 = math.sqrt(sum(g * g for g in genes2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _record_interaction(self, generation: int):
        """记录物种间交互"""
        interaction_record = {
            "generation": generation,
            "species_fitness": [],
            "species_diversity": [],
            "timestamp": time.time()
        }
        
        for population in self.species_populations:
            if population.individuals:
                fitnesses = [ind.fitness for ind in population.individuals]
                interaction_record["species_fitness"].append({
                    "avg": np.mean(fitnesses),
                    "max": max(fitnesses),
                    "min": min(fitnesses)
                })
                interaction_record["species_diversity"].append(population.diversity)
            else:
                interaction_record["species_fitness"].append({"avg": 0, "max": 0, "min": 0})
                interaction_record["species_diversity"].append(0)
        
        self.interaction_history.append(interaction_record)
    
    def _species_migration(self):
        """物种间个体迁移"""
        migration_rate = 0.1  # 10%的个体参与迁移
        
        for i in range(self.num_species):
            source_population = self.species_populations[i]
            target_population = self.species_populations[(i + 1) % self.num_species]
            
            if not source_population.individuals or not target_population.individuals:
                continue
            
            # 选择最佳个体进行迁移
            num_migrants = max(1, int(len(source_population.individuals) * migration_rate))
            source_population.individuals.sort(key=lambda x: x.fitness, reverse=True)
            
            migrants = source_population.individuals[:num_migrants]
            
            # 迁移个体到目标物种
            for migrant in migrants:
                migrant_copy = copy.deepcopy(migrant)
                migrant_copy.metadata["species_id"] = (i + 1) % self.num_species
                migrant_copy.strategy = target_population.individuals[0].strategy
                
                # 替换目标物种中的最差个体
                target_population.individuals.sort(key=lambda x: x.fitness)
                target_population.individuals[0] = migrant_copy
    
    def get_best_solution(self) -> Tuple[Individual, int]:
        """获取所有物种中的最佳解"""
        best_individual = None
        best_species_id = -1
        best_fitness = float('-inf')
        
        for species_id, population in enumerate(self.species_populations):
            if population.individuals:
                species_best = max(population.individuals, key=lambda x: x.fitness)
                if species_best.fitness > best_fitness:
                    best_fitness = species_best.fitness
                    best_individual = species_best
                    best_species_id = species_id
        
        return best_individual, best_species_id
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取协同进化摘要"""
        if not self.interaction_history:
            return {"message": "未开始协同进化"}
        
        latest_record = self.interaction_history[-1]
        best_individual, best_species_id = self.get_best_solution()
        
        return {
            "total_generations": len(self.interaction_history),
            "num_species": self.num_species,
            "best_fitness": best_individual.fitness if best_individual else 0,
            "best_species": best_species_id,
            "latest_generation": latest_record["generation"],
            "species_performance": latest_record["species_fitness"],
            "species_diversity": latest_record["species_diversity"]
        }

class GeneticEvolutionManager:
    """遗传进化管理器"""
    
    def __init__(self):
        self.algorithms = {
            "differential_evolution": DifferentialEvolution(),
            "memetic_algorithm": MemeticAlgorithm(),
            "coevolutionary": CoevolutionaryAlgorithm()
        }
        self.current_algorithm = "differential_evolution"
        self.evolution_history = []
        self.best_solutions = []
    
    def set_algorithm(self, algorithm_name: str):
        """设置当前使用的算法"""
        if algorithm_name in self.algorithms:
            self.current_algorithm = algorithm_name
        else:
            raise ValueError(f"未知算法: {algorithm_name}")
    
    def optimize(self, fitness_func: Callable, dimension: int, 
                population_size: int = 50, generations: int = 100) -> Dict[str, Any]:
        """执行优化"""
        algorithm = self.algorithms[self.current_algorithm]
        
        if isinstance(algorithm, CoevolutionaryAlgorithm):
            # 协同进化算法
            algorithm.initialize_species(dimension)
            algorithm.coevolve(fitness_func, generations)
            best_individual, best_species_id = algorithm.get_best_solution()
            
            result = {
                "algorithm": self.current_algorithm,
                "best_solution": best_individual.genes if best_individual else [],
                "best_fitness": best_individual.fitness if best_individual else 0,
                "generations": generations,
                "summary": algorithm.get_evolution_summary()
            }
        else:
            # 其他算法
            population = algorithm.initialize_population(population_size, dimension)
            
            for generation in range(generations):
                population = algorithm.evolve(population, fitness_func)
                
                # 记录最佳解
                if population.individuals:
                    best_individual = max(population.individuals, key=lambda x: x.fitness)
                    self.best_solutions.append({
                        "generation": generation,
                        "fitness": best_individual.fitness,
                        "genes": best_individual.genes.copy()
                    })
            
            best_individual = max(population.individuals, key=lambda x: x.fitness)
            
            result = {
                "algorithm": self.current_algorithm,
                "best_solution": best_individual.genes,
                "best_fitness": best_individual.fitness,
                "final_population_stats": {
                    "avg_fitness": population.avg_fitness,
                    "best_fitness": population.best_fitness,
                    "diversity": population.diversity,
                    "generation": population.generation
                },
                "evolution_curve": self.best_solutions[-generations:] if self.best_solutions else []
            }
        
        self.evolution_history.append(result)
        return result
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.evolution_history.copy()
    
    def save_results(self, filepath: str):
        """保存结果到文件"""
        results_data = {
            "algorithms": list(self.algorithms.keys()),
            "current_algorithm": self.current_algorithm,
            "evolution_history": self.evolution_history,
            "best_solutions": self.best_solutions,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    def load_results(self, filepath: str):
        """从文件加载结果"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            self.current_algorithm = results_data.get("current_algorithm", "differential_evolution")
            self.evolution_history = results_data.get("evolution_history", [])
            self.best_solutions = results_data.get("best_solutions", [])
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"加载结果失败: {e}")