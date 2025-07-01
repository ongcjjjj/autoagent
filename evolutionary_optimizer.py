"""
Evolutionary Optimization Framework
A comprehensive system for continuous optimization and improvement of evolutionary capabilities
"""

import numpy as np
import random
import time
import threading
import queue
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvolutionMetrics:
    """Metrics for tracking evolutionary progress"""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity: float
    convergence_rate: float
    processing_time: float
    memory_usage: float

@dataclass
class AdaptiveParameters:
    """Self-adapting parameters for evolutionary algorithms"""
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0
    population_size: int = 100
    adaptation_rate: float = 0.01
    
    def adapt(self, performance_improvement: float):
        """Adapt parameters based on performance feedback"""
        if performance_improvement > 0:
            # Good performance, make small adjustments
            self.mutation_rate *= (1 + self.adaptation_rate * performance_improvement)
            self.crossover_rate *= (1 + self.adaptation_rate * performance_improvement * 0.5)
        else:
            # Poor performance, increase exploration
            self.mutation_rate *= (1 + self.adaptation_rate * abs(performance_improvement) * 2)
            self.crossover_rate *= (1 - self.adaptation_rate * abs(performance_improvement) * 0.5)
        
        # Keep parameters within reasonable bounds
        self.mutation_rate = np.clip(self.mutation_rate, 0.01, 0.5)
        self.crossover_rate = np.clip(self.crossover_rate, 0.5, 0.95)

class Individual(ABC):
    """Abstract base class for evolutionary individuals"""
    
    def __init__(self, genes: Any = None):
        self.genes = genes
        self.fitness = None
        self.age = 0
        self.parents = []
        
    @abstractmethod
    def mutate(self, mutation_rate: float):
        """Mutate the individual"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'Individual') -> List['Individual']:
        """Create offspring through crossover"""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, fitness_function: Callable) -> float:
        """Evaluate fitness using the provided function"""
        pass
    
    def __lt__(self, other):
        return self.fitness < other.fitness if self.fitness is not None else False

class RealValuedIndividual(Individual):
    """Individual with real-valued genes"""
    
    def __init__(self, genes: np.ndarray = None, bounds: Tuple[float, float] = (-10, 10)):
        super().__init__(genes)
        self.bounds = bounds
        if genes is None:
            self.genes = np.random.uniform(bounds[0], bounds[1], size=10)
    
    def mutate(self, mutation_rate: float):
        """Gaussian mutation with adaptive step size"""
        if random.random() < mutation_rate:
            mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.1
            noise = np.random.normal(0, mutation_strength, self.genes.shape)
            self.genes = np.clip(self.genes + noise, self.bounds[0], self.bounds[1])
    
    def crossover(self, other: 'RealValuedIndividual') -> List['RealValuedIndividual']:
        """Blend crossover (BLX-α)"""
        alpha = 0.5
        child1_genes = np.zeros_like(self.genes)
        child2_genes = np.zeros_like(self.genes)
        
        for i in range(len(self.genes)):
            min_val = min(self.genes[i], other.genes[i])
            max_val = max(self.genes[i], other.genes[i])
            range_val = max_val - min_val
            
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            child1_genes[i] = np.clip(random.uniform(lower, upper), 
                                    self.bounds[0], self.bounds[1])
            child2_genes[i] = np.clip(random.uniform(lower, upper), 
                                    self.bounds[0], self.bounds[1])
        
        child1 = RealValuedIndividual(child1_genes, self.bounds)
        child2 = RealValuedIndividual(child2_genes, self.bounds)
        child1.parents = [self, other]
        child2.parents = [self, other]
        
        return [child1, child2]
    
    def evaluate_fitness(self, fitness_function: Callable) -> float:
        """Evaluate fitness and cache result"""
        if self.fitness is None:
            self.fitness = fitness_function(self.genes)
        return self.fitness

class MemoryBank:
    """Long-term memory for storing successful strategies and solutions"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.experiences = []
        self.strategy_history = []
        
    def store_experience(self, individual: Individual, generation: int, fitness: float):
        """Store successful individuals for future reference"""
        experience = {
            'genes': individual.genes.copy() if hasattr(individual.genes, 'copy') else individual.genes,
            'fitness': fitness,
            'generation': generation,
            'timestamp': time.time()
        }
        
        self.experiences.append(experience)
        
        # Maintain capacity limit
        if len(self.experiences) > self.capacity:
            self.experiences.sort(key=lambda x: x['fitness'], reverse=True)
            self.experiences = self.experiences[:self.capacity]
    
    def get_elite_experiences(self, top_k: int = 10) -> List[Dict]:
        """Retrieve top performing experiences"""
        sorted_experiences = sorted(self.experiences, key=lambda x: x['fitness'], reverse=True)
        return sorted_experiences[:top_k]
    
    def inject_elite_knowledge(self, population: List[Individual], injection_rate: float = 0.1):
        """Inject elite knowledge into current population"""
        if not self.experiences:
            return
        
        num_inject = int(len(population) * injection_rate)
        elite_experiences = self.get_elite_experiences(num_inject)
        
        for i, experience in enumerate(elite_experiences):
            if i < len(population):
                # Replace worst individuals with elite knowledge
                population[-(i+1)].genes = experience['genes']
                population[-(i+1)].fitness = None  # Force re-evaluation

class IslandModel:
    """Parallel evolution with migration between subpopulations"""
    
    def __init__(self, num_islands: int = 4, migration_rate: float = 0.05):
        self.num_islands = num_islands
        self.migration_rate = migration_rate
        self.islands = []
        self.migration_frequency = 10  # Every N generations
        
    def initialize_islands(self, total_population: int, individual_class: type):
        """Initialize separate island populations"""
        island_size = total_population // self.num_islands
        
        for _ in range(self.num_islands):
            island = [individual_class() for _ in range(island_size)]
            self.islands.append(island)
    
    def migrate(self, generation: int):
        """Migrate best individuals between islands"""
        if generation % self.migration_frequency != 0:
            return
        
        # Get best individuals from each island
        migrants = []
        for island in self.islands:
            island.sort(key=lambda x: x.fitness if x.fitness else float('-inf'), reverse=True)
            num_migrants = max(1, int(len(island) * self.migration_rate))
            migrants.extend(island[:num_migrants])
        
        # Redistribute migrants randomly
        random.shuffle(migrants)
        migrant_idx = 0
        
        for island in self.islands:
            num_to_replace = max(1, int(len(island) * self.migration_rate))
            # Replace worst individuals with migrants
            for i in range(num_to_replace):
                if migrant_idx < len(migrants):
                    island[-(i+1)] = migrants[migrant_idx]
                    migrant_idx += 1

class HybridOptimizer:
    """Combines evolutionary search with local optimization"""
    
    def __init__(self, local_search_frequency: int = 5):
        self.local_search_frequency = local_search_frequency
        
    def local_search(self, individual: RealValuedIndividual, fitness_function: Callable, 
                    max_iterations: int = 10) -> RealValuedIndividual:
        """Simple hill climbing local search"""
        current = individual
        current_fitness = current.evaluate_fitness(fitness_function)
        
        for _ in range(max_iterations):
            # Create neighbor by small perturbation
            neighbor_genes = current.genes + np.random.normal(0, 0.01, current.genes.shape)
            neighbor_genes = np.clip(neighbor_genes, current.bounds[0], current.bounds[1])
            
            neighbor = RealValuedIndividual(neighbor_genes, current.bounds)
            neighbor_fitness = neighbor.evaluate_fitness(fitness_function)
            
            if neighbor_fitness > current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
            else:
                break  # No improvement found
        
        return current

class AdaptiveEvolutionaryOptimizer:
    """Main evolutionary optimization engine with adaptive capabilities"""
    
    def __init__(self, 
                 individual_class: type = RealValuedIndividual,
                 population_size: int = 100,
                 use_parallel: bool = True,
                 use_memory: bool = True,
                 use_islands: bool = True,
                 use_hybrid: bool = True):
        
        self.individual_class = individual_class
        self.population_size = population_size
        self.use_parallel = use_parallel
        self.use_memory = use_memory
        self.use_islands = use_islands
        self.use_hybrid = use_hybrid
        
        # Initialize components
        self.parameters = AdaptiveParameters()
        self.memory = MemoryBank() if use_memory else None
        self.island_model = IslandModel() if use_islands else None
        self.hybrid_optimizer = HybridOptimizer() if use_hybrid else None
        
        # Metrics tracking
        self.metrics_history = []
        self.generation = 0
        
        # Population
        self.population = []
        
    def initialize_population(self):
        """Initialize the population"""
        if self.use_islands and self.island_model:
            self.island_model.initialize_islands(self.population_size, self.individual_class)
            # Flatten for main population reference
            self.population = [ind for island in self.island_model.islands for ind in island]
        else:
            self.population = [self.individual_class() for _ in range(self.population_size)]
    
    def evaluate_population_parallel(self, fitness_function: Callable):
        """Evaluate population fitness in parallel"""
        if self.use_parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(ind.evaluate_fitness, fitness_function) 
                          for ind in self.population]
                for future in futures:
                    future.result()
        else:
            for individual in self.population:
                individual.evaluate_fitness(fitness_function)
    
    def selection(self, tournament_size: int = 3) -> List[Individual]:
        """Tournament selection with adaptive pressure"""
        selected = []
        effective_tournament_size = max(2, int(tournament_size * self.parameters.selection_pressure))
        
        for _ in range(len(self.population)):
            tournament = random.sample(self.population, effective_tournament_size)
            winner = max(tournament, key=lambda x: x.fitness if x.fitness else float('-inf'))
            selected.append(winner)
        
        return selected
    
    def reproduction(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring through crossover and mutation"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            if random.random() < self.parameters.crossover_rate:
                children = parent1.crossover(parent2)
                offspring.extend(children)
            else:
                offspring.extend([parent1, parent2])
        
        # Mutation
        for individual in offspring:
            individual.mutate(self.parameters.mutation_rate)
            individual.age += 1
        
        return offspring[:len(self.population)]
    
    def calculate_diversity(self) -> float:
        """Calculate population diversity (genetic variance)"""
        if not self.population or not hasattr(self.population[0], 'genes'):
            return 0.0
        
        genes_matrix = np.array([ind.genes for ind in self.population])
        return np.mean(np.var(genes_matrix, axis=0))
    
    def update_metrics(self, start_time: float):
        """Update and store performance metrics"""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        
        if fitnesses:
            metrics = EvolutionMetrics(
                generation=self.generation,
                best_fitness=max(fitnesses),
                average_fitness=np.mean(fitnesses),
                diversity=self.calculate_diversity(),
                convergence_rate=0.0,  # TODO: Implement convergence rate calculation
                processing_time=time.time() - start_time,
                memory_usage=0.0  # TODO: Implement memory usage tracking
            )
            
            self.metrics_history.append(metrics)
            
            # Adapt parameters based on performance
            if len(self.metrics_history) > 1:
                performance_improvement = (metrics.best_fitness - 
                                         self.metrics_history[-2].best_fitness)
                self.parameters.adapt(performance_improvement)
    
    def optimize(self, 
                 fitness_function: Callable,
                 max_generations: int = 100,
                 target_fitness: Optional[float] = None,
                 patience: int = 20) -> Tuple[Individual, List[EvolutionMetrics]]:
        """Main optimization loop"""
        
        logger.info("Starting evolutionary optimization...")
        self.initialize_population()
        
        stagnation_counter = 0
        best_fitness_history = []
        
        for gen in range(max_generations):
            start_time = time.time()
            self.generation = gen
            
            # Evaluate population
            self.evaluate_population_parallel(fitness_function)
            
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness if x.fitness else float('-inf'), 
                               reverse=True)
            
            # Update metrics
            self.update_metrics(start_time)
            current_best = self.population[0].fitness
            
            # Store best individuals in memory
            if self.memory and current_best is not None:
                self.memory.store_experience(self.population[0], gen, current_best)
            
            # Apply hybrid local search
            if (self.hybrid_optimizer and 
                gen % self.hybrid_optimizer.local_search_frequency == 0):
                for i in range(min(5, len(self.population))):  # Apply to top 5
                    if isinstance(self.population[i], RealValuedIndividual):
                        self.population[i] = self.hybrid_optimizer.local_search(
                            self.population[i], fitness_function)
            
            # Island migration
            if self.island_model:
                self.island_model.migrate(gen)
            
            # Check termination criteria
            if target_fitness and current_best is not None and current_best >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at generation {gen}")
                break
            
            # Check for stagnation
            best_fitness_history.append(current_best)
            if len(best_fitness_history) > patience:
                recent_improvement = (best_fitness_history[-1] - 
                                    best_fitness_history[-patience])
                if abs(recent_improvement) < 1e-6:
                    stagnation_counter += 1
                    if stagnation_counter >= patience:
                        logger.info(f"Evolution stagnated at generation {gen}")
                        break
                else:
                    stagnation_counter = 0
            
            # Inject elite knowledge periodically
            if self.memory and gen % 20 == 0:
                self.memory.inject_elite_knowledge(self.population)
            
            # Selection and reproduction
            parents = self.selection()
            self.population = self.reproduction(parents)
            
            # Log progress
            if gen % 10 == 0:
                logger.info(f"Generation {gen}: Best fitness = {current_best:.6f}, "
                          f"Avg fitness = {self.metrics_history[-1].average_fitness:.6f}, "
                          f"Diversity = {self.metrics_history[-1].diversity:.6f}")
        
        return self.population[0], self.metrics_history
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        initial_metrics = self.metrics_history[0]
        
        return {
            'total_generations': len(self.metrics_history),
            'final_best_fitness': final_metrics.best_fitness,
            'initial_best_fitness': initial_metrics.best_fitness,
            'improvement': final_metrics.best_fitness - initial_metrics.best_fitness,
            'final_diversity': final_metrics.diversity,
            'average_processing_time': np.mean([m.processing_time for m in self.metrics_history]),
            'convergence_achieved': final_metrics.best_fitness > initial_metrics.best_fitness * 1.1,
            'parameters_used': {
                'mutation_rate': self.parameters.mutation_rate,
                'crossover_rate': self.parameters.crossover_rate,
                'selection_pressure': self.parameters.selection_pressure,
                'population_size': self.population_size
            }
        }

# Example usage and test functions
def sphere_function(x: np.ndarray) -> float:
    """Simple sphere function for testing"""
    return -np.sum(x**2)  # Negative because we maximize

def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function - multimodal test function"""
    A = 10
    n = len(x)
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function - test function with global optimum in narrow valley"""
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def demonstrate_optimization():
    """Demonstrate the evolutionary optimization framework"""
    print("=" * 60)
    print("Evolutionary Optimization Framework Demo")
    print("=" * 60)
    
    # Test different optimization problems
    test_functions = [
        ("Sphere Function", sphere_function),
        ("Rastrigin Function", rastrigin_function),
        ("Rosenbrock Function", rosenbrock_function)
    ]
    
    for func_name, func in test_functions:
        print(f"\nOptimizing {func_name}...")
        print("-" * 40)
        
        optimizer = AdaptiveEvolutionaryOptimizer(
            population_size=50,
            use_parallel=True,
            use_memory=True,
            use_islands=True,
            use_hybrid=True
        )
        
        best_individual, metrics = optimizer.optimize(
            fitness_function=func,
            max_generations=50,
            patience=15
        )
        
        summary = optimizer.get_optimization_summary()
        
        print(f"Best fitness achieved: {best_individual.fitness:.6f}")
        print(f"Best solution: {best_individual.genes[:5]}...")  # Show first 5 genes
        print(f"Total generations: {summary['total_generations']}")
        print(f"Improvement: {summary['improvement']:.6f}")
        print(f"Final diversity: {summary['final_diversity']:.6f}")
        print(f"Average time per generation: {summary['average_processing_time']:.4f}s")

if __name__ == "__main__":
    demonstrate_optimization()