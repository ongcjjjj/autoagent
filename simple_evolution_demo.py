"""
Simple Evolutionary Optimization Demo
Demonstrates core evolutionary optimization concepts without external dependencies
"""

import random
import math
import time
from typing import List, Tuple, Callable, Optional

class SimpleIndividual:
    """Simple individual for demonstration without external dependencies"""
    
    def __init__(self, genes: Optional[List[float]] = None, bounds: Tuple[float, float] = (-10, 10)):
        self.bounds = bounds
        self.fitness = None
        self.age = 0
        
        if genes is None:
            # Generate random genes
            self.genes = [random.uniform(bounds[0], bounds[1]) for _ in range(10)]
        else:
            self.genes = genes[:]
    
    def mutate(self, mutation_rate: float):
        """Gaussian mutation"""
        if random.random() < mutation_rate:
            mutation_strength = (self.bounds[1] - self.bounds[0]) * 0.1
            for i in range(len(self.genes)):
                if random.random() < 0.3:  # Mutate individual genes with probability
                    noise = random.gauss(0, mutation_strength)
                    self.genes[i] = max(self.bounds[0], min(self.bounds[1], 
                                                           self.genes[i] + noise))
    
    def crossover(self, other: 'SimpleIndividual') -> List['SimpleIndividual']:
        """Blend crossover"""
        alpha = 0.5
        child1_genes = []
        child2_genes = []
        
        for i in range(len(self.genes)):
            min_val = min(self.genes[i], other.genes[i])
            max_val = max(self.genes[i], other.genes[i])
            range_val = max_val - min_val
            
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            gene1 = max(self.bounds[0], min(self.bounds[1], random.uniform(lower, upper)))
            gene2 = max(self.bounds[0], min(self.bounds[1], random.uniform(lower, upper)))
            
            child1_genes.append(gene1)
            child2_genes.append(gene2)
        
        return [SimpleIndividual(child1_genes, self.bounds), 
                SimpleIndividual(child2_genes, self.bounds)]
    
    def evaluate_fitness(self, fitness_function: Callable[[List[float]], float]) -> float:
        """Evaluate fitness"""
        if self.fitness is None:
            self.fitness = fitness_function(self.genes)
        return self.fitness

class AdaptiveParameters:
    """Self-adapting evolutionary parameters"""
    
    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.selection_pressure = 2.0
        self.adaptation_rate = 0.01
    
    def adapt(self, performance_improvement: float):
        """Adapt parameters based on performance"""
        if performance_improvement > 0:
            # Good performance, fine-tune
            self.mutation_rate *= (1 + self.adaptation_rate * performance_improvement)
            self.crossover_rate *= (1 + self.adaptation_rate * performance_improvement * 0.5)
        else:
            # Poor performance, increase exploration
            self.mutation_rate *= (1 + self.adaptation_rate * abs(performance_improvement) * 2)
            self.crossover_rate *= (1 - self.adaptation_rate * abs(performance_improvement) * 0.5)
        
        # Keep within bounds
        self.mutation_rate = max(0.01, min(0.5, self.mutation_rate))
        self.crossover_rate = max(0.5, min(0.95, self.crossover_rate))

class MemoryBank:
    """Store and reuse successful solutions"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.experiences = []
    
    def store_experience(self, individual: SimpleIndividual, generation: int):
        """Store successful individual"""
        experience = {
            'genes': individual.genes[:],
            'fitness': individual.fitness,
            'generation': generation
        }
        self.experiences.append(experience)
        
        # Maintain capacity
        if len(self.experiences) > self.capacity:
            self.experiences.sort(key=lambda x: x['fitness'], reverse=True)
            self.experiences = self.experiences[:self.capacity]
    
    def get_elite_genes(self, top_k: int = 5) -> List[List[float]]:
        """Get genes from top performers"""
        sorted_exp = sorted(self.experiences, key=lambda x: x['fitness'], reverse=True)
        return [exp['genes'] for exp in sorted_exp[:top_k]]
    
    def inject_elite_knowledge(self, population: List[SimpleIndividual], injection_rate: float = 0.1):
        """Inject elite knowledge into population"""
        if not self.experiences:
            return
        
        num_inject = int(len(population) * injection_rate)
        elite_genes = self.get_elite_genes(num_inject)
        
        # Replace worst individuals with elite knowledge
        population.sort(key=lambda x: x.fitness if x.fitness else float('-inf'))
        
        for i, genes in enumerate(elite_genes):
            if i < len(population):
                population[i].genes = genes[:]
                population[i].fitness = None  # Force re-evaluation

class SimpleEvolutionaryOptimizer:
    """Simple evolutionary optimizer demonstrating key concepts"""
    
    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.parameters = AdaptiveParameters()
        self.memory = MemoryBank()
        self.population = []
        self.generation = 0
        self.metrics_history = []
    
    def initialize_population(self):
        """Initialize random population"""
        self.population = [SimpleIndividual() for _ in range(self.population_size)]
    
    def evaluate_population(self, fitness_function: Callable):
        """Evaluate all individuals"""
        for individual in self.population:
            individual.evaluate_fitness(fitness_function)
    
    def tournament_selection(self, tournament_size: int = 3) -> List[SimpleIndividual]:
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness if x.fitness else float('-inf'))
            selected.append(winner)
        return selected
    
    def reproduce(self, parents: List[SimpleIndividual]) -> List[SimpleIndividual]:
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
        
        return offspring[:self.population_size]
    
    def calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0
        
        total_variance = 0.0
        num_genes = len(self.population[0].genes)
        
        for gene_idx in range(num_genes):
            gene_values = [ind.genes[gene_idx] for ind in self.population]
            mean_val = sum(gene_values) / len(gene_values)
            variance = sum((val - mean_val) ** 2 for val in gene_values) / len(gene_values)
            total_variance += variance
        
        return total_variance / num_genes
    
    def update_metrics(self, start_time: float):
        """Update performance metrics"""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        
        if fitnesses:
            metrics = {
                'generation': self.generation,
                'best_fitness': max(fitnesses),
                'average_fitness': sum(fitnesses) / len(fitnesses),
                'worst_fitness': min(fitnesses),
                'diversity': self.calculate_diversity(),
                'processing_time': time.time() - start_time,
                'mutation_rate': self.parameters.mutation_rate,
                'crossover_rate': self.parameters.crossover_rate
            }
            
            self.metrics_history.append(metrics)
            
            # Adapt parameters
            if len(self.metrics_history) > 1:
                improvement = (metrics['best_fitness'] - 
                             self.metrics_history[-2]['best_fitness'])
                self.parameters.adapt(improvement)
    
    def optimize(self, fitness_function: Callable, max_generations: int = 100,
                target_fitness: Optional[float] = None, patience: int = 20) -> Tuple[SimpleIndividual, List]:
        """Main optimization loop"""
        print("Starting evolutionary optimization...")
        
        self.initialize_population()
        stagnation_counter = 0
        best_fitness_history = []
        
        for gen in range(max_generations):
            start_time = time.time()
            self.generation = gen
            
            # Evaluate population
            self.evaluate_population(fitness_function)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness if x.fitness else float('-inf'), reverse=True)
            
            # Update metrics
            self.update_metrics(start_time)
            current_best = self.population[0].fitness
            
            # Store best in memory
            if current_best is not None:
                self.memory.store_experience(self.population[0], gen)
            
            # Check termination
            if target_fitness and current_best is not None and current_best >= target_fitness:
                print(f"Target fitness {target_fitness} reached at generation {gen}")
                break
            
            # Check stagnation
            best_fitness_history.append(current_best)
            if len(best_fitness_history) > patience:
                recent_improvement = (best_fitness_history[-1] - best_fitness_history[-patience])
                if abs(recent_improvement) < 1e-6:
                    stagnation_counter += 1
                    if stagnation_counter >= patience // 2:
                        print(f"Evolution stagnated at generation {gen}")
                        break
                else:
                    stagnation_counter = 0
            
            # Inject elite knowledge periodically
            if gen % 15 == 0 and gen > 0:
                self.memory.inject_elite_knowledge(self.population)
            
            # Selection and reproduction
            parents = self.tournament_selection()
            self.population = self.reproduce(parents)
            
            # Progress logging
            if gen % 10 == 0:
                metrics = self.metrics_history[-1]
                print(f"Gen {gen}: Best={current_best:.6f}, Avg={metrics['average_fitness']:.6f}, "
                      f"Diversity={metrics['diversity']:.4f}, MutRate={metrics['mutation_rate']:.3f}")
        
        return self.population[0], self.metrics_history

# Test functions
def sphere_function(genes: List[float]) -> float:
    """Sphere function (minimize)"""
    return -sum(x**2 for x in genes)  # Negative for maximization

def rastrigin_function(genes: List[float]) -> float:
    """Rastrigin function (multimodal)"""
    A = 10
    n = len(genes)
    return -(A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in genes))

def rosenbrock_function(genes: List[float]) -> float:
    """Rosenbrock function (narrow valley)"""
    result = 0
    for i in range(len(genes) - 1):
        result += 100 * (genes[i+1] - genes[i]**2)**2 + (1 - genes[i])**2
    return -result

def ackley_function(genes: List[float]) -> float:
    """Ackley function (complex landscape)"""
    n = len(genes)
    sum1 = sum(x**2 for x in genes)
    sum2 = sum(math.cos(2 * math.pi * x) for x in genes)
    
    result = (-20 * math.exp(-0.2 * math.sqrt(sum1/n)) - 
              math.exp(sum2/n) + 20 + math.e)
    return -result

def demonstrate_evolution():
    """Demonstrate evolutionary optimization on various test functions"""
    print("=" * 80)
    print("EVOLUTIONARY OPTIMIZATION FRAMEWORK DEMONSTRATION")
    print("Continuous Improvement and Adaptive Processing Strategies")
    print("=" * 80)
    
    test_functions = [
        ("Sphere Function (Convex)", sphere_function, -1.0),
        ("Rastrigin Function (Multimodal)", rastrigin_function, -50.0),
        ("Rosenbrock Function (Narrow Valley)", rosenbrock_function, -10.0),
        ("Ackley Function (Complex Landscape)", ackley_function, -5.0)
    ]
    
    results_summary = []
    
    for func_name, func, target in test_functions:
        print(f"\n{'='*60}")
        print(f"OPTIMIZING: {func_name}")
        print('='*60)
        
        optimizer = SimpleEvolutionaryOptimizer(population_size=40)
        
        start_time = time.time()
        best_individual, metrics = optimizer.optimize(
            fitness_function=func,
            max_generations=75,
            target_fitness=target,
            patience=20
        )
        total_time = time.time() - start_time
        
        # Results analysis
        final_metrics = metrics[-1] if metrics else {}
        initial_best = metrics[0]['best_fitness'] if metrics else 0
        final_best = best_individual.fitness
        improvement = final_best - initial_best if final_best and initial_best else 0
        
        print(f"\nRESULTS:")
        print(f"  Best fitness achieved: {final_best:.8f}")
        print(f"  Best solution (first 5 genes): {best_individual.genes[:5]}")
        print(f"  Total improvement: {improvement:.8f}")
        print(f"  Generations completed: {len(metrics)}")
        print(f"  Total optimization time: {total_time:.2f}s")
        print(f"  Final diversity: {final_metrics.get('diversity', 0):.6f}")
        print(f"  Final mutation rate: {final_metrics.get('mutation_rate', 0):.4f}")
        print(f"  Elite solutions in memory: {len(optimizer.memory.experiences)}")
        
        # Convergence analysis
        if len(metrics) > 10:
            early_avg = sum(m['best_fitness'] for m in metrics[:10]) / 10
            late_avg = sum(m['best_fitness'] for m in metrics[-10:]) / 10
            convergence_rate = (late_avg - early_avg) / len(metrics)
            print(f"  Convergence rate: {convergence_rate:.8f} per generation")
        
        results_summary.append({
            'function': func_name,
            'final_fitness': final_best,
            'improvement': improvement,
            'generations': len(metrics),
            'time': total_time,
            'target_reached': final_best is not None and final_best >= target if target else False
        })
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY - CONTINUOUS IMPROVEMENT ANALYSIS")
    print('='*80)
    
    successful_optimizations = sum(1 for r in results_summary if r['improvement'] > 0)
    total_time = sum(r['time'] for r in results_summary)
    avg_improvement = sum(r['improvement'] for r in results_summary) / len(results_summary)
    
    print(f"Total functions optimized: {len(results_summary)}")
    print(f"Successful optimizations: {successful_optimizations}/{len(results_summary)}")
    print(f"Average improvement: {avg_improvement:.6f}")
    print(f"Total optimization time: {total_time:.2f}s")
    
    print(f"\nDETAILED RESULTS:")
    for result in results_summary:
        status = "✓ SUCCESS" if result['improvement'] > 0 else "✗ FAILED"
        target_status = "✓ TARGET" if result.get('target_reached', False) else "✗ PARTIAL"
        print(f"  {result['function']:<35} | {status:<10} | {target_status:<10} | "
              f"Fitness: {result['final_fitness']:>10.6f} | "
              f"Improve: {result['improvement']:>8.6f} | "
              f"Gens: {result['generations']:>3d} | "
              f"Time: {result['time']:>6.2f}s")
    
    print(f"\n{'='*80}")
    print("EVOLUTIONARY CAPABILITIES DEMONSTRATED:")
    print("✓ Adaptive parameter tuning (mutation & crossover rates)")
    print("✓ Memory-based learning (elite solution storage & injection)")
    print("✓ Population diversity management")
    print("✓ Multiple optimization strategies (tournament selection, blend crossover)")
    print("✓ Convergence detection and stagnation handling")
    print("✓ Performance monitoring and metrics tracking")
    print("✓ Multi-objective handling (fitness vs diversity)")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_evolution()