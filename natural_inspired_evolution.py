"""
自然启发进化系统 (Natural-Inspired Evolution System)
整合物理、化学、生物原理的多层次进化架构

基于以下自然原理:
1. 物理原理：引力、电磁场、热力学、量子力学
2. 化学原理：反应动力学、催化作用、分子间作用力
3. 生物原理：遗传进化、群体智能、协同进化
"""

import numpy as np
import json
import random
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import asyncio
from datetime import datetime
import math
import sqlite3

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyType(Enum):
    """能量类型"""
    KINETIC = "kinetic"           # 动能
    POTENTIAL = "potential"       # 势能
    THERMAL = "thermal"          # 热能
    CHEMICAL = "chemical"        # 化学能
    ELECTROMAGNETIC = "electromagnetic"  # 电磁能

class ParticleState(Enum):
    """粒子状态"""
    STABLE = "stable"
    EXCITED = "excited"
    TRANSITION = "transition"
    REACTIVE = "reactive"

class InteractionType(Enum):
    """相互作用类型"""
    GRAVITATIONAL = "gravitational"    # 引力相互作用
    ELECTROMAGNETIC = "electromagnetic" # 电磁相互作用
    STRONG_NUCLEAR = "strong_nuclear"   # 强核力
    WEAK_NUCLEAR = "weak_nuclear"       # 弱核力
    VAN_DER_WAALS = "van_der_waals"    # 范德华力
    HYDROGEN_BOND = "hydrogen_bond"     # 氢键

@dataclass
class Particle:
    """粒子类 - 系统的基本单元"""
    id: str
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    charge: float = 0.0
    spin: float = 0.0
    energy: Dict[EnergyType, float] = field(default_factory=dict)
    state: ParticleState = ParticleState.STABLE
    bonds: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.energy:
            self.energy = {energy_type: 0.0 for energy_type in EnergyType}

@dataclass
class Field:
    """场类 - 描述空间中的物理场"""
    field_type: str
    strength: float
    direction: np.ndarray
    position: np.ndarray
    range_limit: float = float('inf')
    
    def get_force_at(self, position: np.ndarray, particle: Particle) -> np.ndarray:
        """计算粒子在该位置受到的力"""
        distance_vector = position - self.position
        distance = np.linalg.norm(distance_vector)
        
        if distance > self.range_limit:
            return np.zeros_like(position)
            
        if self.field_type == "gravitational":
            # 牛顿万有引力定律
            if distance == 0:
                return np.zeros_like(position)
            force_magnitude = self.strength * particle.mass / (distance ** 2)
            return -force_magnitude * distance_vector / distance
            
        elif self.field_type == "electromagnetic":
            # 库仑力
            if distance == 0:
                return np.zeros_like(position)
            force_magnitude = self.strength * particle.charge / (distance ** 2)
            return force_magnitude * distance_vector / distance
            
        elif self.field_type == "magnetic":
            # 洛伦兹力 F = q(v × B)
            return particle.charge * np.cross(particle.velocity, self.direction)
            
        return np.zeros_like(position)

class PhysicsEngine:
    """物理引擎 - 处理物理相互作用"""
    
    def __init__(self):
        self.fields: List[Field] = []
        self.dt = 0.01  # 时间步长
        self.damping = 0.99  # 阻尼系数
        
    def add_field(self, field: Field):
        """添加物理场"""
        self.fields.append(field)
        
    def calculate_forces(self, particles: List[Particle]) -> Dict[str, np.ndarray]:
        """计算所有粒子受到的合力"""
        forces = {p.id: np.zeros_like(p.position) for p in particles}
        
        # 粒子间相互作用
        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles[i+1:], i+1):
                force = self._calculate_pairwise_force(p1, p2)
                forces[p1.id] += force
                forces[p2.id] -= force  # 牛顿第三定律
                
        # 外场作用
        for particle in particles:
            for field in self.fields:
                field_force = field.get_force_at(particle.position, particle)
                forces[particle.id] += field_force
                
        return forces
        
    def _calculate_pairwise_force(self, p1: Particle, p2: Particle) -> np.ndarray:
        """计算两粒子间的相互作用力"""
        r = p2.position - p1.position
        distance = np.linalg.norm(r)
        
        if distance == 0:
            return np.zeros_like(r)
            
        unit_r = r / distance
        force = np.zeros_like(r)
        
        # 引力相互作用
        G = 6.67e-11  # 引力常数
        gravitational_force = G * p1.mass * p2.mass / (distance ** 2)
        force += gravitational_force * unit_r
        
        # 电磁相互作用
        k = 8.99e9  # 库仑常数
        electromagnetic_force = k * p1.charge * p2.charge / (distance ** 2)
        force += electromagnetic_force * unit_r
        
        # 范德华力（短程相互作用）
        if distance < 10.0:  # 短程作用
            sigma = 1.0
            epsilon = 1.0
            lj_force = 4 * epsilon * (12 * (sigma/distance)**13 - 6 * (sigma/distance)**7)
            force += lj_force * unit_r / distance
            
        return force
        
    def update_particles(self, particles: List[Particle]):
        """更新粒子状态"""
        forces = self.calculate_forces(particles)
        
        for particle in particles:
            force = forces[particle.id]
            
            # 牛顿第二定律 F = ma
            acceleration = force / particle.mass
            
            # 速度更新（Verlet积分）
            particle.velocity += acceleration * self.dt
            particle.velocity *= self.damping  # 阻尼
            
            # 位置更新
            particle.position += particle.velocity * self.dt
            
            # 能量更新
            self._update_particle_energy(particle)
            
    def _update_particle_energy(self, particle: Particle):
        """更新粒子能量"""
        # 动能
        particle.energy[EnergyType.KINETIC] = 0.5 * particle.mass * np.dot(particle.velocity, particle.velocity)
        
        # 热能（基于速度的随机成分）
        thermal_factor = np.random.normal(1.0, 0.1)
        particle.energy[EnergyType.THERMAL] = particle.energy[EnergyType.KINETIC] * thermal_factor

class ChemicalReactor:
    """化学反应器 - 处理化学反应"""
    
    def __init__(self):
        self.reactions: List[Dict] = []
        self.catalysts: Dict[str, float] = {}  # 催化剂及其效率
        self.temperature = 300.0  # 开尔文
        self.pressure = 1.0  # 大气压
        
    def add_reaction(self, reactants: List[str], products: List[str], 
                    activation_energy: float, rate_constant: float):
        """添加化学反应"""
        self.reactions.append({
            'reactants': reactants,
            'products': products,
            'activation_energy': activation_energy,
            'rate_constant': rate_constant
        })
        
    def add_catalyst(self, catalyst_id: str, efficiency: float):
        """添加催化剂"""
        self.catalysts[catalyst_id] = efficiency
        
    def process_reactions(self, particles: List[Particle]) -> List[Particle]:
        """处理化学反应"""
        new_particles = []
        reacted_particles = set()
        
        for reaction in self.reactions:
            # 查找反应物
            reactant_particles = self._find_reactants(particles, reaction['reactants'])
            
            if not reactant_particles:
                continue
                
            # 计算反应速率
            rate = self._calculate_reaction_rate(reaction, reactant_particles)
            
            # 判断是否发生反应
            if random.random() < rate:
                # 生成产物
                products = self._create_products(reaction['products'], reactant_particles)
                new_particles.extend(products)
                
                # 标记反应物为已反应
                for rp in reactant_particles:
                    reacted_particles.add(rp.id)
                    
        # 移除已反应的粒子
        remaining_particles = [p for p in particles if p.id not in reacted_particles]
        
        return remaining_particles + new_particles
        
    def _find_reactants(self, particles: List[Particle], reactant_types: List[str]) -> List[Particle]:
        """查找反应物"""
        found_reactants = []
        available_particles = particles.copy()
        
        for reactant_type in reactant_types:
            for particle in available_particles:
                if particle.properties.get('type') == reactant_type:
                    found_reactants.append(particle)
                    available_particles.remove(particle)
                    break
            else:
                return []  # 找不到足够的反应物
                
        return found_reactants
        
    def _calculate_reaction_rate(self, reaction: Dict, reactants: List[Particle]) -> float:
        """计算反应速率（阿伦尼乌斯方程）"""
        # 基础速率常数
        k = reaction['rate_constant']
        
        # 温度影响（阿伦尼乌斯方程）
        R = 8.314  # 气体常数
        Ea = reaction['activation_energy']
        k_temp = k * np.exp(-Ea / (R * self.temperature))
        
        # 催化剂影响
        catalyst_factor = 1.0
        for particle in reactants:
            if particle.id in self.catalysts:
                catalyst_factor *= (1 + self.catalysts[particle.id])
                
        # 浓度影响（反应物浓度的乘积）
        concentration_factor = 1.0
        for particle in reactants:
            concentration_factor *= particle.properties.get('concentration', 1.0)
            
        return k_temp * catalyst_factor * concentration_factor
        
    def _create_products(self, product_types: List[str], reactants: List[Particle]) -> List[Particle]:
        """创建反应产物"""
        products = []
        
        # 计算质心位置和总动量
        total_mass = sum(p.mass for p in reactants)
        center_of_mass = sum(p.mass * p.position for p in reactants) / total_mass
        total_momentum = sum(p.mass * p.velocity for p in reactants)
        
        for i, product_type in enumerate(product_types):
            # 创建产物粒子
            product = Particle(
                id=f"product_{product_type}_{random.randint(1000, 9999)}",
                position=center_of_mass + np.random.normal(0, 0.1, 3),
                velocity=total_momentum / len(product_types) / 1.0,  # 假设产物质量
                mass=1.0,  # 假设产物质量
                properties={'type': product_type, 'generation': 0}
            )
            
            # 设置能量（考虑反应热）
            product.energy[EnergyType.CHEMICAL] = sum(
                r.energy.get(EnergyType.CHEMICAL, 0) for r in reactants
            ) / len(product_types)
            
            products.append(product)
            
        return products

class BiologicalEvolution:
    """生物进化系统"""
    
    def __init__(self):
        self.population: List[Particle] = []
        self.generation = 0
        self.mutation_rate = 0.01
        self.crossover_rate = 0.7
        self.selection_pressure = 2.0
        self.environment_pressure = 1.0
        
    def initialize_population(self, size: int) -> List[Particle]:
        """初始化种群"""
        self.population = []
        
        for i in range(size):
            # 创建个体（生物粒子）
            individual = Particle(
                id=f"organism_{i}",
                position=np.random.uniform(-10, 10, 3),
                velocity=np.random.uniform(-1, 1, 3),
                mass=np.random.uniform(0.5, 2.0),
                charge=np.random.uniform(-1, 1),
                properties={
                    'genome': self._generate_random_genome(),
                    'fitness': 0.0,
                    'age': 0,
                    'generation': self.generation,
                    'species': 'base_organism'
                }
            )
            
            self.population.append(individual)
            
        return self.population
        
    def _generate_random_genome(self, length: int = 50) -> List[float]:
        """生成随机基因组"""
        return [random.uniform(0, 1) for _ in range(length)]
        
    def evaluate_fitness(self, environment_data: Dict = None) -> None:
        """评估适应度"""
        for individual in self.population:
            fitness = 0.0
            
            # 基于能量的适应度
            total_energy = sum(individual.energy.values())
            fitness += total_energy * 0.3
            
            # 基于位置的适应度（距离原点的距离）
            distance_fitness = 1.0 / (1.0 + np.linalg.norm(individual.position))
            fitness += distance_fitness * 0.2
            
            # 基于基因组的适应度
            genome = individual.properties.get('genome', [])
            if genome:
                genome_fitness = sum(genome) / len(genome)
                fitness += genome_fitness * 0.3
                
            # 环境适应度
            if environment_data:
                env_fitness = self._calculate_environment_fitness(individual, environment_data)
                fitness += env_fitness * 0.2
                
            individual.properties['fitness'] = fitness
            
    def _calculate_environment_fitness(self, individual: Particle, env_data: Dict) -> float:
        """计算环境适应度"""
        fitness = 0.0
        
        # 温度适应性
        optimal_temp = env_data.get('optimal_temperature', 300.0)
        current_temp = env_data.get('current_temperature', 300.0)
        temp_diff = abs(current_temp - optimal_temp)
        temp_fitness = np.exp(-temp_diff / 50.0)  # 高斯适应曲线
        fitness += temp_fitness * 0.5
        
        # 资源可用性
        resources = env_data.get('resources', 1.0)
        resource_fitness = min(resources, 1.0)
        fitness += resource_fitness * 0.3
        
        # 竞争压力
        competition = env_data.get('competition', 0.5)
        competition_fitness = 1.0 - competition
        fitness += competition_fitness * 0.2
        
        return fitness
        
    def selection(self, selection_size: int = None) -> List[Particle]:
        """选择操作（锦标赛选择）"""
        if selection_size is None:
            selection_size = len(self.population) // 2
            
        selected = []
        
        for _ in range(selection_size):
            # 锦标赛选择
            tournament_size = max(2, int(self.selection_pressure))
            candidates = random.sample(self.population, tournament_size)
            winner = max(candidates, key=lambda x: x.properties.get('fitness', 0))
            selected.append(winner)
            
        return selected
        
    def crossover(self, parent1: Particle, parent2: Particle) -> Tuple[Particle, Particle]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        # 基因组交叉
        genome1 = parent1.properties.get('genome', []).copy()
        genome2 = parent2.properties.get('genome', []).copy()
        
        if genome1 and genome2 and len(genome1) == len(genome2):
            # 单点交叉
            crossover_point = random.randint(1, len(genome1) - 1)
            new_genome1 = genome1[:crossover_point] + genome2[crossover_point:]
            new_genome2 = genome2[:crossover_point] + genome1[crossover_point:]
        else:
            new_genome1, new_genome2 = genome1, genome2
            
        # 创建子代
        child1 = Particle(
            id=f"child_{random.randint(1000, 9999)}",
            position=(parent1.position + parent2.position) / 2 + np.random.normal(0, 0.1, 3),
            velocity=(parent1.velocity + parent2.velocity) / 2,
            mass=(parent1.mass + parent2.mass) / 2,
            charge=(parent1.charge + parent2.charge) / 2,
            properties={
                'genome': new_genome1,
                'fitness': 0.0,
                'age': 0,
                'generation': self.generation + 1,
                'species': parent1.properties.get('species', 'unknown')
            }
        )
        
        child2 = Particle(
            id=f"child_{random.randint(1000, 9999)}",
            position=(parent1.position + parent2.position) / 2 + np.random.normal(0, 0.1, 3),
            velocity=(parent1.velocity + parent2.velocity) / 2,
            mass=(parent1.mass + parent2.mass) / 2,
            charge=(parent1.charge + parent2.charge) / 2,
            properties={
                'genome': new_genome2,
                'fitness': 0.0,
                'age': 0,
                'generation': self.generation + 1,
                'species': parent2.properties.get('species', 'unknown')
            }
        )
        
        return child1, child2
        
    def mutation(self, individual: Particle) -> Particle:
        """变异操作"""
        if random.random() > self.mutation_rate:
            return individual
            
        # 基因组变异
        genome = individual.properties.get('genome', []).copy()
        if genome:
            for i in range(len(genome)):
                if random.random() < self.mutation_rate:
                    genome[i] += np.random.normal(0, 0.1)
                    genome[i] = max(0, min(1, genome[i]))  # 保持在[0,1]范围内
                    
        # 物理属性变异
        if random.random() < self.mutation_rate:
            individual.mass *= np.random.normal(1.0, 0.05)
            individual.mass = max(0.1, individual.mass)
            
        if random.random() < self.mutation_rate:
            individual.charge += np.random.normal(0, 0.1)
            individual.charge = max(-2, min(2, individual.charge))
            
        # 位置变异
        if random.random() < self.mutation_rate:
            individual.position += np.random.normal(0, 0.5, 3)
            
        individual.properties['genome'] = genome
        return individual
        
    def evolve_generation(self, environment_data: Dict = None) -> List[Particle]:
        """进化一代"""
        # 评估适应度
        self.evaluate_fitness(environment_data)
        
        # 选择
        selected = self.selection()
        
        # 生成新种群
        new_population = []
        
        # 精英保留
        elite_size = max(1, len(self.population) // 10)
        elite = sorted(self.population, key=lambda x: x.properties.get('fitness', 0), reverse=True)[:elite_size]
        new_population.extend(elite)
        
        # 交叉和变异生成剩余个体
        while len(new_population) < len(self.population):
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_population.extend([child1, child2])
            
        # 保持种群大小
        self.population = new_population[:len(self.population)]
        self.generation += 1
        
        return self.population

class SwarmIntelligence:
    """群体智能系统"""
    
    def __init__(self):
        self.swarms: Dict[str, List[Particle]] = {}
        self.pheromone_map: Dict[Tuple, float] = defaultdict(float)
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
    def create_swarm(self, swarm_id: str, particles: List[Particle]):
        """创建群体"""
        self.swarms[swarm_id] = particles
        
        # 初始化粒子群优化参数
        for particle in particles:
            particle.properties.update({
                'swarm_id': swarm_id,
                'personal_best_position': particle.position.copy(),
                'personal_best_fitness': 0.0,
                'velocity_inertia': 0.9,
                'cognitive_weight': 2.0,
                'social_weight': 2.0
            })
            
    def update_swarm_pso(self, swarm_id: str, objective_function=None):
        """粒子群优化更新"""
        if swarm_id not in self.swarms:
            return
            
        swarm = self.swarms[swarm_id]
        
        for particle in swarm:
            # 评估适应度
            if objective_function:
                fitness = objective_function(particle)
            else:
                fitness = -np.linalg.norm(particle.position)  # 默认目标：接近原点
                
            # 更新个体最优
            if fitness > particle.properties['personal_best_fitness']:
                particle.properties['personal_best_fitness'] = fitness
                particle.properties['personal_best_position'] = particle.position.copy()
                
            # 更新全局最优
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
                
        # 更新速度和位置
        for particle in swarm:
            if self.global_best_position is not None:
                # PSO速度更新公式
                w = particle.properties['velocity_inertia']
                c1 = particle.properties['cognitive_weight']
                c2 = particle.properties['social_weight']
                
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = c1 * r1 * (
                    particle.properties['personal_best_position'] - particle.position
                )
                social_velocity = c2 * r2 * (
                    self.global_best_position - particle.position
                )
                
                particle.velocity = (w * particle.velocity + 
                                   cognitive_velocity + social_velocity)
                
                # 位置更新
                particle.position += particle.velocity
                
    def update_swarm_aco(self, swarm_id: str, target_positions: List[np.ndarray]):
        """蚁群优化更新"""
        if swarm_id not in self.swarms:
            return
            
        swarm = self.swarms[swarm_id]
        evaporation_rate = 0.1
        
        # 信息素蒸发
        for pos_key in list(self.pheromone_map.keys()):
            self.pheromone_map[pos_key] *= (1 - evaporation_rate)
            if self.pheromone_map[pos_key] < 0.01:
                del self.pheromone_map[pos_key]
                
        # 蚂蚁移动和信息素更新
        for particle in swarm:
            # 选择下一个位置（基于信息素浓度和启发式信息）
            best_target = None
            best_attractiveness = 0
            
            for target in target_positions:
                distance = np.linalg.norm(target - particle.position)
                if distance > 0:
                    # 信息素浓度
                    pos_key = tuple(target.astype(int))
                    pheromone = self.pheromone_map[pos_key]
                    
                    # 启发式信息（距离的倒数）
                    heuristic = 1.0 / distance
                    
                    # 总吸引力
                    attractiveness = (pheromone ** 1.0) * (heuristic ** 2.0)
                    
                    if attractiveness > best_attractiveness:
                        best_attractiveness = attractiveness
                        best_target = target
                        
            # 移动向目标
            if best_target is not None:
                direction = best_target - particle.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    particle.velocity = 0.1 * direction / distance
                    particle.position += particle.velocity
                    
                    # 留下信息素
                    pos_key = tuple(particle.position.astype(int))
                    self.pheromone_map[pos_key] += 1.0
                    
    def update_swarm_flocking(self, swarm_id: str):
        """群体行为更新（鸟群/鱼群）"""
        if swarm_id not in self.swarms:
            return
            
        swarm = self.swarms[swarm_id]
        separation_radius = 2.0
        alignment_radius = 5.0
        cohesion_radius = 8.0
        
        for particle in swarm:
            separation = np.zeros(3)
            alignment = np.zeros(3)
            cohesion = np.zeros(3)
            
            sep_count = 0
            align_count = 0
            coh_count = 0
            
            for other in swarm:
                if other.id == particle.id:
                    continue
                    
                distance = np.linalg.norm(other.position - particle.position)
                
                # 分离行为
                if distance < separation_radius and distance > 0:
                    diff = particle.position - other.position
                    separation += diff / distance
                    sep_count += 1
                    
                # 对齐行为
                if distance < alignment_radius:
                    alignment += other.velocity
                    align_count += 1
                    
                # 聚集行为
                if distance < cohesion_radius:
                    cohesion += other.position
                    coh_count += 1
                    
            # 归一化行为向量
            if sep_count > 0:
                separation /= sep_count
                separation = separation / np.linalg.norm(separation) if np.linalg.norm(separation) > 0 else separation
                
            if align_count > 0:
                alignment /= align_count
                alignment = alignment / np.linalg.norm(alignment) if np.linalg.norm(alignment) > 0 else alignment
                
            if coh_count > 0:
                cohesion /= coh_count
                cohesion = cohesion - particle.position
                cohesion = cohesion / np.linalg.norm(cohesion) if np.linalg.norm(cohesion) > 0 else cohesion
                
            # 综合行为
            total_force = 1.5 * separation + 1.0 * alignment + 1.0 * cohesion
            
            # 限制最大速度
            max_speed = 2.0
            if np.linalg.norm(total_force) > max_speed:
                total_force = total_force / np.linalg.norm(total_force) * max_speed
                
            particle.velocity = 0.8 * particle.velocity + 0.2 * total_force

class NaturalInspiredEvolutionSystem:
    """自然启发进化系统主类"""
    
    def __init__(self, config_file: str = None):
        self.physics_engine = PhysicsEngine()
        self.chemical_reactor = ChemicalReactor()
        self.biological_evolution = BiologicalEvolution()
        self.swarm_intelligence = SwarmIntelligence()
        
        self.particles: List[Particle] = []
        self.environment_data: Dict = {
            'current_temperature': 300.0,
            'optimal_temperature': 298.0,
            'resources': 1.0,
            'competition': 0.3,
            'time_step': 0
        }
        
        self.simulation_time = 0
        self.max_time = 1000
        self.config = self._load_config(config_file) if config_file else {}
        
        # 初始化数据库
        self._init_database()
        
    def _init_database(self):
        """初始化数据库"""
        self.conn = sqlite3.connect('natural_evolution.db')
        cursor = self.conn.cursor()
        
        # 创建粒子历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS particle_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                particle_id TEXT,
                time_step INTEGER,
                position_x REAL,
                position_y REAL,
                position_z REAL,
                velocity_x REAL,
                velocity_y REAL,
                velocity_z REAL,
                mass REAL,
                charge REAL,
                fitness REAL,
                generation INTEGER,
                species TEXT
            )
        ''')
        
        # 创建环境历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_step INTEGER,
                temperature REAL,
                resources REAL,
                competition REAL,
                total_particles INTEGER,
                average_fitness REAL
            )
        ''')
        
        self.conn.commit()
        
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"无法加载配置文件 {config_file}: {e}")
            return {}
            
    def initialize_system(self, num_particles: int = 100):
        """初始化系统"""
        logger.info(f"初始化自然启发进化系统，粒子数量: {num_particles}")
        
        # 初始化粒子
        self.particles = self.biological_evolution.initialize_population(num_particles)
        
        # 添加物理场
        self._setup_physical_fields()
        
        # 添加化学反应
        self._setup_chemical_reactions()
        
        # 创建群体
        self._setup_swarms()
        
        logger.info("系统初始化完成")
        
    def _setup_physical_fields(self):
        """设置物理场"""
        # 添加重力场
        gravity_field = Field(
            field_type="gravitational",
            strength=9.8,
            direction=np.array([0, 0, -1]),
            position=np.array([0, 0, 0]),
            range_limit=100.0
        )
        self.physics_engine.add_field(gravity_field)
        
        # 添加电磁场
        em_field = Field(
            field_type="electromagnetic",
            strength=1.0,
            direction=np.array([1, 0, 0]),
            position=np.array([0, 0, 0]),
            range_limit=50.0
        )
        self.physics_engine.add_field(em_field)
        
    def _setup_chemical_reactions(self):
        """设置化学反应"""
        # 添加基本反应
        self.chemical_reactor.add_reaction(
            reactants=['A', 'B'],
            products=['C'],
            activation_energy=50.0,
            rate_constant=0.1
        )
        
        self.chemical_reactor.add_reaction(
            reactants=['C', 'D'],
            products=['E', 'F'],
            activation_energy=30.0,
            rate_constant=0.2
        )
        
        # 添加催化剂
        self.chemical_reactor.add_catalyst('catalyst_1', 2.0)
        
    def _setup_swarms(self):
        """设置群体"""
        # 将粒子分成不同的群体
        swarm_size = len(self.particles) // 3
        
        # PSO群体
        pso_swarm = self.particles[:swarm_size]
        self.swarm_intelligence.create_swarm('pso_swarm', pso_swarm)
        
        # ACO群体
        aco_swarm = self.particles[swarm_size:2*swarm_size]
        self.swarm_intelligence.create_swarm('aco_swarm', aco_swarm)
        
        # 鸟群/鱼群
        flocking_swarm = self.particles[2*swarm_size:]
        self.swarm_intelligence.create_swarm('flocking_swarm', flocking_swarm)
        
    def update_environment(self):
        """更新环境参数"""
        time_factor = self.simulation_time / self.max_time
        
        # 温度变化（模拟季节性变化）
        base_temp = 300.0
        temp_variation = 20.0 * np.sin(2 * np.pi * time_factor * 4)  # 4个周期
        self.environment_data['current_temperature'] = base_temp + temp_variation
        
        # 资源变化
        self.environment_data['resources'] = 0.5 + 0.5 * np.sin(2 * np.pi * time_factor * 2)
        
        # 竞争压力变化
        self.environment_data['competition'] = 0.2 + 0.6 * (1 - np.exp(-time_factor * 3))
        
        # 更新时间步
        self.environment_data['time_step'] = self.simulation_time
        
    def run_simulation_step(self):
        """运行一个仿真步骤"""
        # 更新环境
        self.update_environment()
        
        # 物理引擎更新
        self.physics_engine.update_particles(self.particles)
        
        # 化学反应处理
        self.particles = self.chemical_reactor.process_reactions(self.particles)
        
        # 群体智能更新
        self.swarm_intelligence.update_swarm_pso('pso_swarm')
        
        # ACO更新需要目标位置
        target_positions = [np.array([0, 0, 0])]  # 简单目标：原点
        self.swarm_intelligence.update_swarm_aco('aco_swarm', target_positions)
        
        self.swarm_intelligence.update_swarm_flocking('flocking_swarm')
        
        # 生物进化（每10步进化一次）
        if self.simulation_time % 10 == 0:
            self.particles = self.biological_evolution.evolve_generation(self.environment_data)
            
        # 记录数据
        self._record_data()
        
        self.simulation_time += 1
        
    def _record_data(self):
        """记录仿真数据"""
        cursor = self.conn.cursor()
        
        # 记录粒子状态
        for particle in self.particles:
            cursor.execute('''
                INSERT INTO particle_history 
                (particle_id, time_step, position_x, position_y, position_z,
                 velocity_x, velocity_y, velocity_z, mass, charge, fitness, generation, species)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                particle.id, self.simulation_time,
                particle.position[0], particle.position[1], particle.position[2],
                particle.velocity[0], particle.velocity[1], particle.velocity[2],
                particle.mass, particle.charge,
                particle.properties.get('fitness', 0.0),
                particle.properties.get('generation', 0),
                particle.properties.get('species', 'unknown')
            ))
            
        # 记录环境状态
        avg_fitness = np.mean([p.properties.get('fitness', 0.0) for p in self.particles])
        cursor.execute('''
            INSERT INTO environment_history 
            (time_step, temperature, resources, competition, total_particles, average_fitness)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.simulation_time,
            self.environment_data['current_temperature'],
            self.environment_data['resources'],
            self.environment_data['competition'],
            len(self.particles),
            avg_fitness
        ))
        
        self.conn.commit()
        
    def run_simulation(self, max_steps: int = None):
        """运行完整仿真"""
        if max_steps:
            self.max_time = max_steps
            
        logger.info(f"开始仿真，最大步数: {self.max_time}")
        
        try:
            while self.simulation_time < self.max_time:
                self.run_simulation_step()
                
                # 每100步输出一次进度
                if self.simulation_time % 100 == 0:
                    avg_fitness = np.mean([p.properties.get('fitness', 0.0) for p in self.particles])
                    logger.info(f"步骤 {self.simulation_time}/{self.max_time}, "
                              f"粒子数量: {len(self.particles)}, "
                              f"平均适应度: {avg_fitness:.4f}, "
                              f"温度: {self.environment_data['current_temperature']:.2f}K")
                              
        except KeyboardInterrupt:
            logger.info("仿真被用户中断")
        except Exception as e:
            logger.error(f"仿真过程中发生错误: {e}")
        finally:
            logger.info("仿真完成")
            
    def get_simulation_statistics(self) -> Dict:
        """获取仿真统计信息"""
        cursor = self.conn.cursor()
        
        # 获取最终统计
        cursor.execute('''
            SELECT AVG(fitness), MAX(fitness), MIN(fitness), COUNT(DISTINCT particle_id)
            FROM particle_history WHERE time_step = ?
        ''', (self.simulation_time - 1,))
        
        result = cursor.fetchone()
        avg_fitness, max_fitness, min_fitness, num_particles = result
        
        # 获取进化统计
        cursor.execute('''
            SELECT MAX(generation), AVG(generation)
            FROM particle_history WHERE time_step = ?
        ''', (self.simulation_time - 1,))
        
        gen_result = cursor.fetchone()
        max_generation, avg_generation = gen_result
        
        # 获取环境统计
        cursor.execute('''
            SELECT AVG(temperature), AVG(resources), AVG(competition)
            FROM environment_history
        ''')
        
        env_result = cursor.fetchone()
        avg_temp, avg_resources, avg_competition = env_result
        
        return {
            'simulation_steps': self.simulation_time,
            'final_particle_count': num_particles,
            'fitness_statistics': {
                'average': avg_fitness,
                'maximum': max_fitness,
                'minimum': min_fitness
            },
            'evolution_statistics': {
                'max_generation': max_generation,
                'average_generation': avg_generation
            },
            'environment_statistics': {
                'average_temperature': avg_temp,
                'average_resources': avg_resources,
                'average_competition': avg_competition
            }
        }
        
    def save_results(self, filename: str):
        """保存仿真结果"""
        stats = self.get_simulation_statistics()
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.info(f"仿真结果已保存到 {filename}")
        
    def close(self):
        """关闭系统"""
        if hasattr(self, 'conn'):
            self.conn.close()
        logger.info("系统已关闭")

# 示例使用
if __name__ == "__main__":
    # 创建自然启发进化系统
    system = NaturalInspiredEvolutionSystem()
    
    try:
        # 初始化系统
        system.initialize_system(num_particles=150)
        
        # 运行仿真
        system.run_simulation(max_steps=500)
        
        # 获取统计信息
        stats = system.get_simulation_statistics()
        print("\n=== 仿真统计信息 ===")
        print(f"仿真步数: {stats['simulation_steps']}")
        print(f"最终粒子数量: {stats['final_particle_count']}")
        print(f"适应度统计: 平均={stats['fitness_statistics']['average']:.4f}, "
              f"最大={stats['fitness_statistics']['maximum']:.4f}, "
              f"最小={stats['fitness_statistics']['minimum']:.4f}")
        print(f"进化统计: 最大代数={stats['evolution_statistics']['max_generation']}, "
              f"平均代数={stats['evolution_statistics']['average_generation']:.2f}")
        
        # 保存结果
        system.save_results("natural_evolution_results.json")
        
    finally:
        system.close()