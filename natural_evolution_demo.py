"""
自然启发进化系统演示程序
整合物理、化学、生物原理的多层次进化演示
"""

import sys
import os
import time
import random
import logging
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    logger.warning("NumPy未安装，将使用简化的数学运算")
    HAS_NUMPY = False
    # 简化的numpy替代
    class SimpleArray:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = list(data)
            else:
                self.data = [data]
        
        def __add__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a + b for a, b in zip(self.data, other.data)])
            return SimpleArray([x + other for x in self.data])
        
        def __mul__(self, other):
            if isinstance(other, (int, float)):
                return SimpleArray([x * other for x in self.data])
            return SimpleArray([a * b for a, b in zip(self.data, other.data)])
        
        def __truediv__(self, other):
            if isinstance(other, (int, float)):
                return SimpleArray([x / other for x in self.data])
            return SimpleArray([a / b for a, b in zip(self.data, other.data)])
        
        def __getitem__(self, index):
            return self.data[index]
        
        def __setitem__(self, index, value):
            self.data[index] = value
        
        def norm(self):
            return (sum(x**2 for x in self.data))**0.5
        
        def copy(self):
            return SimpleArray(self.data.copy())
    
    # 替代numpy函数
    def array(data):
        return SimpleArray(data)
    
    def zeros(size):
        return SimpleArray([0.0] * size)
    
    def random_uniform(low, high, size):
        return SimpleArray([random.uniform(low, high) for _ in range(size)])
    
    def random_normal(mean, std, size):
        return SimpleArray([random.gauss(mean, std) for _ in range(size)])
    
    def linalg_norm(arr):
        return arr.norm()
    
    def dot(a, b):
        return sum(x * y for x, y in zip(a.data, b.data))
    
    def mean(arr_list):
        if not arr_list:
            return 0.0
        return sum(arr_list) / len(arr_list)
    
    def sin(x):
        import math
        return math.sin(x)
    
    def exp(x):
        import math
        return math.exp(x)
    
    def pi():
        import math
        return math.pi
    
    # 将这些函数放入一个模拟的numpy模块
    class MockNumpy:
        array = staticmethod(array)
        zeros = staticmethod(zeros)
        random = type('random', (), {
            'uniform': lambda low, high, size: random_uniform(low, high, size),
            'normal': lambda mean, std, size: random_normal(mean, std, size)
        })()
        linalg = type('linalg', (), {'norm': staticmethod(linalg_norm)})()
        dot = staticmethod(dot)
        mean = staticmethod(mean)
        sin = staticmethod(sin)
        exp = staticmethod(exp)
        pi = pi()
    
    np = MockNumpy()

class SimpleEvolutionDemo:
    """简化的自然启发进化演示"""
    
    def __init__(self):
        self.population = []
        self.generation = 0
        self.population_size = 50
        self.genome_length = 20
        self.mutation_rate = 0.05
        self.crossover_rate = 0.8
        self.environment_temperature = 300.0
        self.environment_resources = 1.0
        
        # 物理参数
        self.gravity_strength = 0.1
        self.electromagnetic_strength = 0.05
        
        # 化学参数
        self.reaction_rate = 0.02
        self.catalyst_efficiency = 1.5
        
        # 群体智能参数
        self.swarm_cohesion = 0.1
        self.swarm_separation = 0.2
        self.swarm_alignment = 0.15
        
        logger.info("简化进化系统初始化完成")
    
    def create_individual(self, individual_id: str) -> Dict:
        """创建个体"""
        if HAS_NUMPY:
            position = np.random.uniform(-10, 10, 3)
            velocity = np.random.uniform(-1, 1, 3)
            genome = [random.random() for _ in range(self.genome_length)]
        else:
            position = [random.uniform(-10, 10) for _ in range(3)]
            velocity = [random.uniform(-1, 1) for _ in range(3)]
            genome = [random.random() for _ in range(self.genome_length)]
        
        return {
            'id': individual_id,
            'position': position,
            'velocity': velocity,
            'mass': random.uniform(0.5, 2.0),
            'charge': random.uniform(-1, 1),
            'genome': genome,
            'fitness': 0.0,
            'generation': self.generation,
            'energy': {
                'kinetic': 0.0,
                'potential': 0.0,
                'chemical': random.uniform(0, 10)
            },
            'species': 'base_organism',
            'age': 0
        }
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for i in range(self.population_size):
            individual = self.create_individual(f"org_{i}")
            self.population.append(individual)
        
        logger.info(f"初始化种群完成，个体数量: {len(self.population)}")
    
    def calculate_fitness(self, individual: Dict) -> float:
        """计算适应度"""
        fitness = 0.0
        
        # 1. 物理适应度（基于位置和能量）
        if HAS_NUMPY:
            position_norm = np.linalg.norm(individual['position'])
        else:
            position_norm = sum(x**2 for x in individual['position'])**0.5
        
        # 距离原点越近，适应度越高
        position_fitness = 1.0 / (1.0 + position_norm)
        fitness += position_fitness * 0.3
        
        # 能量适应度
        total_energy = sum(individual['energy'].values())
        energy_fitness = min(total_energy / 10.0, 1.0)  # 归一化到[0,1]
        fitness += energy_fitness * 0.2
        
        # 2. 化学适应度（基于基因组）
        genome_sum = sum(individual['genome'])
        genome_fitness = genome_sum / len(individual['genome'])
        fitness += genome_fitness * 0.3
        
        # 3. 生物适应度（环境适应性）
        # 温度适应性
        optimal_temp = 298.0
        temp_diff = abs(self.environment_temperature - optimal_temp)
        if HAS_NUMPY:
            temp_fitness = np.exp(-temp_diff / 50.0)
        else:
            import math
            temp_fitness = math.exp(-temp_diff / 50.0)
        fitness += temp_fitness * 0.1
        
        # 资源利用效率
        resource_fitness = min(self.environment_resources, 1.0)
        fitness += resource_fitness * 0.1
        
        return fitness
    
    def evaluate_population(self):
        """评估整个种群的适应度"""
        for individual in self.population:
            individual['fitness'] = self.calculate_fitness(individual)
    
    def physics_update(self, individual: Dict):
        """物理更新（牛顿力学）"""
        # 计算重力影响
        if HAS_NUMPY:
            gravity_force = np.array([0, 0, -self.gravity_strength * individual['mass']])
            individual['velocity'] = np.array(individual['velocity']) + gravity_force * 0.01
            
            # 电磁力影响（基于电荷）
            em_force = np.array([self.electromagnetic_strength * individual['charge'], 0, 0])
            individual['velocity'] += em_force * 0.01
            
            # 位置更新
            individual['position'] = np.array(individual['position']) + np.array(individual['velocity']) * 0.01
            
            # 阻尼
            individual['velocity'] *= 0.99
            
            # 更新动能
            velocity_magnitude = np.linalg.norm(individual['velocity'])
            individual['energy']['kinetic'] = 0.5 * individual['mass'] * velocity_magnitude**2
        else:
            # 简化版本
            gravity_force = [0, 0, -self.gravity_strength * individual['mass']]
            for i in range(3):
                individual['velocity'][i] += gravity_force[i] * 0.01
            
            # 电磁力
            em_force = [self.electromagnetic_strength * individual['charge'], 0, 0]
            individual['velocity'][0] += em_force[0] * 0.01
            
            # 位置更新
            for i in range(3):
                individual['position'][i] += individual['velocity'][i] * 0.01
                individual['velocity'][i] *= 0.99  # 阻尼
            
            # 动能
            velocity_magnitude = sum(v**2 for v in individual['velocity'])**0.5
            individual['energy']['kinetic'] = 0.5 * individual['mass'] * velocity_magnitude**2
    
    def chemical_reactions(self):
        """化学反应处理"""
        # 简化的化学反应：高能量个体可能产生新个体
        new_individuals = []
        
        for individual in self.population:
            if individual['energy']['chemical'] > 8.0 and random.random() < self.reaction_rate:
                # 发生"化学反应"，产生新个体
                new_individual = self.create_individual(f"reaction_{random.randint(1000, 9999)}")
                
                # 继承部分特征
                new_individual['position'] = individual['position'].copy() if hasattr(individual['position'], 'copy') else individual['position'][:]
                new_individual['mass'] = individual['mass'] * 0.8
                new_individual['genome'] = individual['genome'][:self.genome_length//2] + [random.random() for _ in range(self.genome_length//2)]
                
                # 能量重新分配
                individual['energy']['chemical'] *= 0.6
                new_individual['energy']['chemical'] = individual['energy']['chemical'] * 0.4
                
                new_individuals.append(new_individual)
        
        # 添加新个体到种群
        self.population.extend(new_individuals)
        
        if new_individuals:
            logger.info(f"化学反应产生了 {len(new_individuals)} 个新个体")
    
    def swarm_behavior(self):
        """群体行为更新"""
        for individual in self.population:
            if HAS_NUMPY:
                separation = np.zeros(3)
                alignment = np.zeros(3)
                cohesion = np.zeros(3)
            else:
                separation = [0.0, 0.0, 0.0]
                alignment = [0.0, 0.0, 0.0]
                cohesion = [0.0, 0.0, 0.0]
            
            neighbor_count = 0
            
            for other in self.population:
                if other['id'] == individual['id']:
                    continue
                
                # 计算距离
                if HAS_NUMPY:
                    distance = np.linalg.norm(np.array(other['position']) - np.array(individual['position']))
                else:
                    distance = sum((a-b)**2 for a, b in zip(other['position'], individual['position']))**0.5
                
                if distance < 5.0:  # 邻居半径
                    neighbor_count += 1
                    
                    # 分离行为
                    if distance > 0 and distance < 2.0:
                        if HAS_NUMPY:
                            diff = np.array(individual['position']) - np.array(other['position'])
                            separation += diff / distance
                        else:
                            for i in range(3):
                                diff = individual['position'][i] - other['position'][i]
                                separation[i] += diff / distance
                    
                    # 对齐行为
                    if HAS_NUMPY:
                        alignment += np.array(other['velocity'])
                    else:
                        for i in range(3):
                            alignment[i] += other['velocity'][i]
                    
                    # 聚集行为
                    if HAS_NUMPY:
                        cohesion += np.array(other['position'])
                    else:
                        for i in range(3):
                            cohesion[i] += other['position'][i]
            
            if neighbor_count > 0:
                if HAS_NUMPY:
                    alignment /= neighbor_count
                    cohesion = cohesion / neighbor_count - np.array(individual['position'])
                    
                    # 应用群体行为
                    total_force = (self.swarm_separation * separation + 
                                 self.swarm_alignment * alignment + 
                                 self.swarm_cohesion * cohesion)
                    
                    individual['velocity'] = np.array(individual['velocity']) + total_force * 0.01
                else:
                    for i in range(3):
                        alignment[i] /= neighbor_count
                        cohesion[i] = cohesion[i] / neighbor_count - individual['position'][i]
                        
                        # 应用群体行为
                        total_force = (self.swarm_separation * separation[i] + 
                                     self.swarm_alignment * alignment[i] + 
                                     self.swarm_cohesion * cohesion[i])
                        
                        individual['velocity'][i] += total_force * 0.01
    
    def selection(self) -> List[Dict]:
        """选择操作（锦标赛选择）"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(self.population) // 2):
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(candidates, key=lambda x: x['fitness'])
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> tuple:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # 创建子代
        child1 = self.create_individual(f"child_{random.randint(1000, 9999)}")
        child2 = self.create_individual(f"child_{random.randint(1000, 9999)}")
        
        # 基因组交叉
        crossover_point = random.randint(1, self.genome_length - 1)
        child1['genome'] = parent1['genome'][:crossover_point] + parent2['genome'][crossover_point:]
        child2['genome'] = parent2['genome'][:crossover_point] + parent1['genome'][crossover_point:]
        
        # 物理属性交叉
        child1['mass'] = (parent1['mass'] + parent2['mass']) / 2
        child2['mass'] = (parent1['mass'] + parent2['mass']) / 2
        
        child1['charge'] = (parent1['charge'] + parent2['charge']) / 2
        child2['charge'] = (parent1['charge'] + parent2['charge']) / 2
        
        # 位置继承（在父母中间附近）
        if HAS_NUMPY:
            center = (np.array(parent1['position']) + np.array(parent2['position'])) / 2
            child1['position'] = center + np.random.normal(0, 0.1, 3)
            child2['position'] = center + np.random.normal(0, 0.1, 3)
        else:
            center = [(p1 + p2) / 2 for p1, p2 in zip(parent1['position'], parent2['position'])]
            child1['position'] = [c + random.gauss(0, 0.1) for c in center]
            child2['position'] = [c + random.gauss(0, 0.1) for c in center]
        
        child1['generation'] = self.generation + 1
        child2['generation'] = self.generation + 1
        
        return child1, child2
    
    def mutation(self, individual: Dict):
        """变异操作"""
        # 基因组变异
        for i in range(len(individual['genome'])):
            if random.random() < self.mutation_rate:
                individual['genome'][i] += random.gauss(0, 0.1)
                individual['genome'][i] = max(0, min(1, individual['genome'][i]))
        
        # 物理属性变异
        if random.random() < self.mutation_rate:
            individual['mass'] *= random.uniform(0.9, 1.1)
            individual['mass'] = max(0.1, individual['mass'])
        
        if random.random() < self.mutation_rate:
            individual['charge'] += random.gauss(0, 0.1)
            individual['charge'] = max(-2, min(2, individual['charge']))
        
        # 位置变异
        if random.random() < self.mutation_rate:
            if HAS_NUMPY:
                individual['position'] += np.random.normal(0, 0.5, 3)
            else:
                for i in range(3):
                    individual['position'][i] += random.gauss(0, 0.5)
    
    def evolve_generation(self):
        """进化一代"""
        # 评估适应度
        self.evaluate_population()
        
        # 选择
        selected = self.selection()
        
        # 生成新种群
        new_population = []
        
        # 精英保留
        elite_size = max(1, len(self.population) // 10)
        elite = sorted(self.population, key=lambda x: x['fitness'], reverse=True)[:elite_size]
        new_population.extend(elite)
        
        # 交叉和变异
        while len(new_population) < self.population_size:
            if len(selected) >= 2:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutation(child1)
                self.mutation(child2)
                
                new_population.extend([child1, child2])
            else:
                # 如果选择的个体不足，直接复制
                if selected:
                    new_individual = selected[0].copy()
                    self.mutation(new_individual)
                    new_population.append(new_individual)
        
        # 保持种群大小
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def update_environment(self, step: int):
        """更新环境参数"""
        # 温度周期性变化
        if HAS_NUMPY:
            temp_variation = 20.0 * np.sin(2 * np.pi * step / 100)
        else:
            import math
            temp_variation = 20.0 * math.sin(2 * math.pi * step / 100)
        
        self.environment_temperature = 300.0 + temp_variation
        
        # 资源周期性变化
        if HAS_NUMPY:
            self.environment_resources = 0.5 + 0.5 * np.sin(2 * np.pi * step / 50)
        else:
            import math
            self.environment_resources = 0.5 + 0.5 * math.sin(2 * math.pi * step / 50)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.population:
            return {}
        
        fitness_values = [ind['fitness'] for ind in self.population]
        generations = [ind['generation'] for ind in self.population]
        masses = [ind['mass'] for ind in self.population]
        charges = [ind['charge'] for ind in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'fitness': {
                'average': sum(fitness_values) / len(fitness_values),
                'maximum': max(fitness_values),
                'minimum': min(fitness_values)
            },
            'mass': {
                'average': sum(masses) / len(masses),
                'range': [min(masses), max(masses)]
            },
            'charge': {
                'average': sum(charges) / len(charges),
                'range': [min(charges), max(charges)]
            },
            'generation_range': [min(generations), max(generations)],
            'environment': {
                'temperature': self.environment_temperature,
                'resources': self.environment_resources
            }
        }
    
    def run_simulation(self, max_generations: int = 100, max_steps: int = 1000):
        """运行仿真"""
        logger.info(f"开始自然启发进化仿真 - 最大代数: {max_generations}, 最大步数: {max_steps}")
        
        # 初始化种群
        self.initialize_population()
        
        step = 0
        last_evolution = 0
        
        try:
            while step < max_steps and self.generation < max_generations:
                # 更新环境
                self.update_environment(step)
                
                # 物理更新
                for individual in self.population:
                    self.physics_update(individual)
                
                # 化学反应
                if step % 10 == 0:  # 每10步处理一次化学反应
                    self.chemical_reactions()
                
                # 群体行为
                self.swarm_behavior()
                
                # 生物进化（每20步进化一次）
                if step - last_evolution >= 20:
                    self.evolve_generation()
                    last_evolution = step
                
                # 输出统计信息
                if step % 100 == 0:
                    stats = self.get_statistics()
                    logger.info(f"步骤 {step}/{max_steps} - "
                              f"代数: {stats['generation']}, "
                              f"种群大小: {stats['population_size']}, "
                              f"平均适应度: {stats['fitness']['average']:.4f}, "
                              f"最大适应度: {stats['fitness']['maximum']:.4f}, "
                              f"环境温度: {stats['environment']['temperature']:.2f}K")
                
                step += 1
                
                # 控制仿真速度
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("仿真被用户中断")
        except Exception as e:
            logger.error(f"仿真过程中发生错误: {e}")
        
        # 最终统计
        final_stats = self.get_statistics()
        logger.info("\n=== 仿真完成 ===")
        logger.info(f"总步数: {step}")
        logger.info(f"最终代数: {final_stats['generation']}")
        logger.info(f"最终种群大小: {final_stats['population_size']}")
        logger.info(f"最终平均适应度: {final_stats['fitness']['average']:.4f}")
        logger.info(f"最大适应度: {final_stats['fitness']['maximum']:.4f}")
        logger.info(f"质量范围: {final_stats['mass']['range']}")
        logger.info(f"电荷范围: {final_stats['charge']['range']}")
        
        return final_stats

def main():
    """主函数"""
    print("=" * 60)
    print("自然启发进化系统演示")
    print("整合物理、化学、生物原理的多层次进化架构")
    print("=" * 60)
    
    if not HAS_NUMPY:
        print("注意：NumPy未安装，使用简化数学运算")
        print("建议安装NumPy以获得更好的性能：pip install numpy")
        print("-" * 60)
    
    # 创建演示系统
    demo = SimpleEvolutionDemo()
    
    try:
        # 运行仿真
        stats = demo.run_simulation(max_generations=50, max_steps=800)
        
        # 保存结果
        import json
        result_file = "natural_evolution_demo_results.json"
        with open(result_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n结果已保存到: {result_file}")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()