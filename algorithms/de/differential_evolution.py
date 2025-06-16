import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import random


class DifferentialEvolution:
    def __init__(self,
                 objective_function: Callable[[np.ndarray], float],
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 F: float = 0.8,
                 CR: float = 0.9,
                 # {rand} random method to select target vector, {1} one differance vector, {bin} binomial crossover
                 strategy: str = "DE/rand/1/bin",
                 patience: int = 30
                 ):
        """
        Parameters:
        - objective_function: Function to minimize f(x) -> float
        - bounds: List of (min, max) tuples for each dimension
        - population_size: Number of individuals in population
        - F: Scale factor (typically 0.4-0.9)
        - CR: Crossover probability (typically 0.8-0.95) [0.1, 1]
        - strategy: DE strategy (currently supports "DE/rand/1/bin")
        - seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.F = F  # Scale factor
        self.CR = CR  # Crossover probability
        self.strategy = strategy
        self.patience = patience

        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.diversity_history = []
        self.generation = 0
        self.stagnation_count = 0

    def initialize_population(self) -> None:
        """
        Initialize population randomly within bounds.
        Each individual is a vector of decision variables.
        """
        self.population = np.zeros((self.population_size, self.dimensions))

        for i in range(self.population_size):
            for j in range(self.dimensions):
                min_bound, max_bound = self.bounds[j]
                self.population[i, j] = np.random.uniform(min_bound, max_bound)

        self.evaluate_population()

        initial_diversity = self.calculate_population_diversity()
        self.diversity_history.append(initial_diversity)

    def calculate_population_diversity(self) -> float:
        if self.population is None or self.population_size < 2:
            return 0.0
        distances = []
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                distances.append(np.linalg.norm(self.population[i] - self.population[j]))

        return float(np.mean(distances))

    def evaluate_population(self) -> None:
        self.fitness = np.zeros(self.population_size)

        for i in range(self.population_size):
            self.fitness[i] = self.objective_function(self.population[i])

            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_individual = self.population[i].copy()

    def mutation(self, target_index: int) -> np.ndarray:
        """
        Create mutant vector using DE/rand/1 strategy.

        Formula: v_i = x_r1 + F * (x_r2 - x_r3)
        where r1, r2, r3 are random distinct indices != target_index

        Parameters:
        - target_index: Index of current target individual

        Returns:
        - mutant_vector: New mutant vector
        """
        candidates = list(range(self.population_size))
        candidates.remove(target_index)

        r1, r2, r3 = random.sample(candidates, 3)

        # v = x_r1 + F * (x_r2 - x_r3)
        mutant_vector = (self.population[r1] +
                         self.F * (self.population[r2] - self.population[r3]))

        # ensure that mutant vector is within bounds
        mutant_vector = self.repair_bounds(mutant_vector)

        return mutant_vector

    def crossover(self, target_vector: np.ndarray, mutant_vector: np.ndarray) -> np.ndarray:
        trial_vector = target_vector.copy()

        j_rand = np.random.randint(0, self.dimensions)

        for j in range(self.dimensions):
            if np.random.random() < self.CR or j == j_rand:
                trial_vector[j] = mutant_vector[j]

        return trial_vector

    def selection(self, target_vector: np.ndarray, trial_vector: np.ndarray,
                  target_fitness: float) -> Tuple[np.ndarray, float]:
        trial_fitness = self.objective_function(trial_vector)

        # Select the better individual (minimization)
        if trial_fitness < target_fitness:
            return trial_vector, trial_fitness
        else:
            return target_vector, target_fitness

    def repair_bounds(self, vector: np.ndarray) -> np.ndarray:
        """
        Repair vector to ensure all values are within bounds.
        Uses boundary constraint handling.
        """
        repaired_vector = vector.copy()

        for i in range(self.dimensions):
            min_bound, max_bound = self.bounds[i]
            repaired_vector[i] = np.clip(repaired_vector[i], min_bound, max_bound)

        return repaired_vector

    def evolve_generation(self) -> None:
        """
        Perform one generation of DE evolution.
        For each individual: mutation -> crossover -> selection
        """
        new_population = np.zeros_like(self.population)
        new_fitness = np.zeros_like(self.fitness)

        for i in range(self.population_size):
            mutant_vector = self.mutation(i)

            trial_vector = self.crossover(self.population[i], mutant_vector)

            selected_vector, selected_fitness = self.selection(
                self.population[i], trial_vector, self.fitness[i]
            )

            new_population[i] = selected_vector
            new_fitness[i] = selected_fitness

            if selected_fitness < self.best_fitness:
                self.best_fitness = selected_fitness
                self.best_individual = selected_vector.copy()

        # Update population and fitness
        self.population = new_population
        self.fitness = new_fitness
        self.generation += 1

        # Track fitness history
        self.fitness_history.append(self.best_fitness)
        current_diversity = self.calculate_population_diversity()
        self.diversity_history.append(current_diversity)

    def optimize(self, max_generations: int = 1000,
                 tolerance: float = 1e-8,
                 verbose: bool = True) -> dict:
        """
        Run the complete DE optimization process.

        Parameters:
        - max_generations: Maximum number of generations
        - tolerance: Convergence tolerance
        - verbose: Print progress information

        Returns:
        - result: Dictionary with optimization results
        """
        # Initialize population if not already done
        if self.population is None:
            self.initialize_population()

        print(f"\nStarting DE optimization...")
        print(f"Strategy: {self.strategy}")
        print(f"Population size: {self.population_size}")
        print(f"F (mutation factor): {self.F}")
        print(f"CR (crossover rate): {self.CR}")
        print(f"Dimensions: {self.dimensions}")
        print("-" * 50)

        # Evolution loop
        for generation in range(max_generations):
            prev_best = self.best_fitness

            # Evolve one generation
            self.evolve_generation()

            # Print progress
            if verbose and (generation + 1) % 100 == 0:
                print(f"Generation {generation + 1:4d}: "
                      f"Best fitness = {self.best_fitness:.8f}")

            # Check convergence
            if abs(prev_best - self.best_fitness) < tolerance:
                print(f"\nprev_best = {prev_best:.8f}, best_fitness = {self.best_fitness:.8f}")
                if verbose:
                    print(f"\nConverged at generation {generation + 1}")
                if self.stagnation_count >= self.patience:
                    print(f"Stagnation detected after {self.patience} generations without improvement.")
                    break
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0

        # Prepare results
        result = {
            'best_solution': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations': self.generation,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'final_population': self.population,
            'final_fitness': self.fitness
        }

        if verbose:
            print(f"\nOptimization completed!")
            print(f"Best solution: {self.best_individual}")
            print(f"Best fitness: {self.best_fitness:.8f}")
            print(f"Total generations: {self.generation}")

        return result

    def visualize_results(self) -> None:
        if not self.diversity_history or not self.fitness_history:
            print("No data available. Run optimization first.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        generations = range(len(self.diversity_history))

        # Plot diversity
        ax1.plot(generations, self.diversity_history, 'b-', linewidth=2, label='Population Diversity')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Diversity')
        ax1.set_title('Population Diversity Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot fitness
        fitness_generations = range(len(self.fitness_history))
        ax2.plot(fitness_generations, self.fitness_history, 'r-', linewidth=2, label='Best Fitness')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('Fitness Evolution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Print insights
        print(f"\nDiversity Analysis:")
        print(f"Initial diversity: {self.diversity_history[0]:.4f}")
        print(f"Final diversity: {self.diversity_history[-1]:.4f}")
        print(f"Diversity reduction: {self.diversity_history[0] / self.diversity_history[-1]:.2f}x")
