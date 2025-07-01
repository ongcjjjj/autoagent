"""
Neural Network Evolution Module
Advanced evolutionary strategies for neural network optimization and architecture search
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from evolutionary_optimizer import Individual, AdaptiveEvolutionaryOptimizer
import json
import time

@dataclass
class NetworkArchitecture:
    """Represents a neural network architecture"""
    layers: List[int]  # Number of neurons in each layer
    activations: List[str]  # Activation functions for each layer
    dropout_rates: List[float]  # Dropout rates for each layer
    learning_rate: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layers': self.layers,
            'activations': self.activations,
            'dropout_rates': self.dropout_rates,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

class SimpleNeuralNetwork:
    """Simple neural network implementation for demonstration"""
    
    def __init__(self, architecture: NetworkArchitecture):
        self.architecture = architecture
        self.weights = []
        self.biases = []
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        layers = self.architecture.layers
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            fan_in, fan_out = layers[i], layers[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            weight_matrix = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias_vector = np.zeros(fan_out)
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def activation_function(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function"""
        if activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'linear':
            return x
        else:
            return x  # Default to linear
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        current_input = X
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            current_input = np.dot(current_input, weight) + bias
            
            # Apply activation function
            if i < len(self.architecture.activations):
                activation = self.architecture.activations[i]
                current_input = self.activation_function(current_input, activation)
        
        return current_input
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X)

class NeuralNetworkIndividual(Individual):
    """Individual representing a neural network"""
    
    def __init__(self, architecture: Optional[NetworkArchitecture] = None, 
                 weight_bounds: Tuple[float, float] = (-2.0, 2.0)):
        super().__init__()
        self.weight_bounds = weight_bounds
        
        if architecture is None:
            # Generate random architecture
            self.architecture = self.generate_random_architecture()
        else:
            self.architecture = architecture
        
        # Initialize the neural network
        self.network = SimpleNeuralNetwork(self.architecture)
        
        # Flatten weights for evolutionary operations
        self.genes = self.flatten_weights()
    
    def generate_random_architecture(self) -> NetworkArchitecture:
        """Generate a random neural network architecture"""
        # Random number of hidden layers (1-3)
        num_hidden_layers = random.randint(1, 3)
        
        # Input and output sizes (fixed for this example)
        input_size = 10
        output_size = 1
        
        # Random hidden layer sizes
        hidden_sizes = [random.randint(5, 50) for _ in range(num_hidden_layers)]
        layers = [input_size] + hidden_sizes + [output_size]
        
        # Random activation functions
        activation_choices = ['relu', 'sigmoid', 'tanh']
        activations = [random.choice(activation_choices) for _ in range(len(layers) - 2)]
        activations.append('linear')  # Linear output for regression
        
        # Random dropout rates
        dropout_rates = [random.uniform(0.0, 0.3) for _ in range(len(layers) - 1)]
        
        # Random hyperparameters
        learning_rate = random.uniform(0.001, 0.1)
        batch_size = random.choice([16, 32, 64, 128])
        
        return NetworkArchitecture(
            layers=layers,
            activations=activations,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
    
    def flatten_weights(self) -> np.ndarray:
        """Flatten all weights and biases into a single vector"""
        flattened = []
        
        for weight_matrix in self.network.weights:
            flattened.extend(weight_matrix.flatten())
        
        for bias_vector in self.network.biases:
            flattened.extend(bias_vector.flatten())
        
        return np.array(flattened)
    
    def unflatten_weights(self, flattened_weights: np.ndarray):
        """Reconstruct weight matrices from flattened vector"""
        self.network.weights = []
        self.network.biases = []
        
        idx = 0
        layers = self.architecture.layers
        
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            
            # Reconstruct weight matrix
            weight_size = fan_in * fan_out
            weight_data = flattened_weights[idx:idx + weight_size]
            weight_matrix = weight_data.reshape(fan_in, fan_out)
            self.network.weights.append(weight_matrix)
            idx += weight_size
            
            # Reconstruct bias vector
            bias_data = flattened_weights[idx:idx + fan_out]
            self.network.biases.append(bias_data)
            idx += fan_out
    
    def mutate(self, mutation_rate: float):
        """Mutate network weights and potentially architecture"""
        # Weight mutation
        if random.random() < mutation_rate:
            mutation_strength = 0.1
            noise = np.random.normal(0, mutation_strength, self.genes.shape)
            self.genes = np.clip(self.genes + noise, 
                               self.weight_bounds[0], self.weight_bounds[1])
            self.unflatten_weights(self.genes)
        
        # Architecture mutation (less frequent)
        if random.random() < mutation_rate * 0.1:
            self.mutate_architecture()
    
    def mutate_architecture(self):
        """Mutate the network architecture"""
        mutation_type = random.choice(['add_neuron', 'remove_neuron', 'change_activation'])
        
        if mutation_type == 'add_neuron' and len(self.architecture.layers) > 2:
            # Add a neuron to a random hidden layer
            layer_idx = random.randint(1, len(self.architecture.layers) - 2)
            if self.architecture.layers[layer_idx] < 100:  # Limit size
                self.architecture.layers[layer_idx] += 1
                self.rebuild_network()
        
        elif mutation_type == 'remove_neuron' and len(self.architecture.layers) > 2:
            # Remove a neuron from a random hidden layer
            layer_idx = random.randint(1, len(self.architecture.layers) - 2)
            if self.architecture.layers[layer_idx] > 2:  # Minimum size
                self.architecture.layers[layer_idx] -= 1
                self.rebuild_network()
        
        elif mutation_type == 'change_activation':
            # Change activation function
            if len(self.architecture.activations) > 0:
                act_idx = random.randint(0, len(self.architecture.activations) - 2)
                activation_choices = ['relu', 'sigmoid', 'tanh']
                self.architecture.activations[act_idx] = random.choice(activation_choices)
    
    def rebuild_network(self):
        """Rebuild network after architecture mutation"""
        self.network = SimpleNeuralNetwork(self.architecture)
        self.genes = self.flatten_weights()
    
    def crossover(self, other: 'NeuralNetworkIndividual') -> List['NeuralNetworkIndividual']:
        """Create offspring through architecture and weight crossover"""
        child1 = NeuralNetworkIndividual()
        child2 = NeuralNetworkIndividual()
        
        # Architecture crossover - blend architectures
        child1.architecture = self.crossover_architecture(self.architecture, other.architecture)
        child2.architecture = self.crossover_architecture(other.architecture, self.architecture)
        
        # Rebuild networks with new architectures
        child1.rebuild_network()
        child2.rebuild_network()
        
        # Weight crossover - if architectures are compatible
        if len(self.genes) == len(other.genes):
            # Uniform crossover
            crossover_mask = np.random.random(len(self.genes)) < 0.5
            
            child1_genes = np.where(crossover_mask, self.genes, other.genes)
            child2_genes = np.where(crossover_mask, other.genes, self.genes)
            
            child1.genes = child1_genes
            child2.genes = child2_genes
            
            child1.unflatten_weights(child1.genes)
            child2.unflatten_weights(child2.genes)
        
        child1.parents = [self, other]
        child2.parents = [self, other]
        
        return [child1, child2]
    
    def crossover_architecture(self, arch1: NetworkArchitecture, 
                             arch2: NetworkArchitecture) -> NetworkArchitecture:
        """Crossover two architectures"""
        # Take average layer sizes where possible
        max_layers = max(len(arch1.layers), len(arch2.layers))
        new_layers = []
        
        for i in range(max_layers):
            if i < len(arch1.layers) and i < len(arch2.layers):
                if i == 0 or i == max_layers - 1:
                    # Keep input/output sizes fixed
                    new_layers.append(arch1.layers[i])
                else:
                    # Average hidden layer sizes
                    avg_size = int((arch1.layers[i] + arch2.layers[i]) / 2)
                    new_layers.append(max(2, avg_size))  # Minimum 2 neurons
            elif i < len(arch1.layers):
                new_layers.append(arch1.layers[i])
            else:
                new_layers.append(arch2.layers[i])
        
        # Crossover other parameters
        new_activations = arch1.activations if random.random() < 0.5 else arch2.activations
        new_dropout_rates = arch1.dropout_rates if random.random() < 0.5 else arch2.dropout_rates
        new_learning_rate = (arch1.learning_rate + arch2.learning_rate) / 2
        new_batch_size = arch1.batch_size if random.random() < 0.5 else arch2.batch_size
        
        # Adjust lists to match new layer count
        while len(new_activations) < len(new_layers) - 1:
            new_activations.append('relu')
        while len(new_dropout_rates) < len(new_layers) - 1:
            new_dropout_rates.append(0.1)
        
        new_activations = new_activations[:len(new_layers) - 1]
        new_dropout_rates = new_dropout_rates[:len(new_layers) - 1]
        
        return NetworkArchitecture(
            layers=new_layers,
            activations=new_activations,
            dropout_rates=new_dropout_rates,
            learning_rate=new_learning_rate,
            batch_size=new_batch_size
        )
    
    def evaluate_fitness(self, fitness_function) -> float:
        """Evaluate network performance"""
        if self.fitness is None:
            self.fitness = fitness_function(self.network)
        return self.fitness

class NeuroEvolutionOptimizer(AdaptiveEvolutionaryOptimizer):
    """Specialized optimizer for neural network evolution"""
    
    def __init__(self, population_size: int = 50, **kwargs):
        super().__init__(
            individual_class=NeuralNetworkIndividual,
            population_size=population_size,
            **kwargs
        )
        self.architecture_diversity_weight = 0.2
    
    def calculate_architecture_diversity(self) -> float:
        """Calculate diversity in network architectures"""
        if not self.population:
            return 0.0
        
        architectures = []
        for individual in self.population:
            if hasattr(individual, 'architecture'):
                arch_str = str(individual.architecture.layers)
                architectures.append(arch_str)
        
        unique_architectures = len(set(architectures))
        return unique_architectures / len(self.population)
    
    def calculate_diversity(self) -> float:
        """Enhanced diversity calculation including architecture diversity"""
        weight_diversity = super().calculate_diversity()
        arch_diversity = self.calculate_architecture_diversity()
        
        return (weight_diversity * (1 - self.architecture_diversity_weight) + 
                arch_diversity * self.architecture_diversity_weight)

def generate_regression_data(n_samples: int = 1000, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data for testing"""
    X = np.random.uniform(-1, 1, (n_samples, 10))
    
    # Complex function with interactions
    y = (np.sin(X[:, 0] * 3) + 
         np.cos(X[:, 1] * 2) * X[:, 2] + 
         X[:, 3] * X[:, 4] - 
         0.5 * X[:, 5]**2 + 
         noise * np.random.normal(0, 1, n_samples))
    
    return X, y.reshape(-1, 1)

def neural_network_fitness(network: SimpleNeuralNetwork, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fitness function for neural networks based on validation performance"""
    try:
        # Training predictions
        train_pred = network.predict(X_train)
        train_mse = np.mean((train_pred - y_train)**2)
        
        # Validation predictions
        val_pred = network.predict(X_val)
        val_mse = np.mean((val_pred - y_val)**2)
        
        # Fitness combines validation performance and model complexity
        complexity_penalty = len(network.weights) * 0.001  # Penalize complexity
        
        # Return negative MSE (since we maximize fitness)
        fitness = -val_mse - complexity_penalty
        
        # Penalize overfitting
        if val_mse > train_mse * 2:
            fitness -= (val_mse - train_mse)
        
        return fitness
    
    except Exception as e:
        # Return very low fitness for problematic networks
        return -1000.0

def demonstrate_neural_evolution():
    """Demonstrate neural network evolution"""
    print("=" * 60)
    print("Neural Network Evolution Demo")
    print("=" * 60)
    
    # Generate data
    print("Generating synthetic regression data...")
    X, y = generate_regression_data(n_samples=1000, noise=0.1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create fitness function
    def fitness_func(network):
        return neural_network_fitness(network, X_train, y_train, X_val, y_val)
    
    # Run evolution
    print("\nStarting neural network evolution...")
    optimizer = NeuroEvolutionOptimizer(
        population_size=30,
        use_parallel=True,
        use_memory=True,
        use_islands=False,  # Disabled for neural networks due to complexity
        use_hybrid=False    # Disabled for neural networks
    )
    
    best_network_individual, metrics = optimizer.optimize(
        fitness_function=fitness_func,
        max_generations=50,
        patience=15
    )
    
    # Results
    print("\n" + "=" * 50)
    print("Evolution Results")
    print("=" * 50)
    
    best_network = best_network_individual.network
    best_arch = best_network_individual.architecture
    
    print(f"Best fitness achieved: {best_network_individual.fitness:.6f}")
    print(f"Best architecture: {best_arch.layers}")
    print(f"Activations: {best_arch.activations}")
    print(f"Learning rate: {best_arch.learning_rate:.4f}")
    print(f"Batch size: {best_arch.batch_size}")
    
    # Test performance
    val_pred = best_network.predict(X_val)
    val_mse = np.mean((val_pred - y_val)**2)
    print(f"Final validation MSE: {val_mse:.6f}")
    
    # Architecture diversity analysis
    arch_diversity = optimizer.calculate_architecture_diversity()
    print(f"Final architecture diversity: {arch_diversity:.3f}")
    
    summary = optimizer.get_optimization_summary()
    print(f"Total generations: {summary['total_generations']}")
    print(f"Improvement: {summary['improvement']:.6f}")
    
    return best_network_individual, metrics

if __name__ == "__main__":
    demonstrate_neural_evolution()