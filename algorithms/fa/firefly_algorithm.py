from typing import Callable, Tuple

import numpy as np
from matplotlib import pyplot as plt

from algorithms.fa.firefly import Firefly


class FireflyAlgorithm:
    def __init__(self,
                 objective_function: Callable[[np.ndarray], float],
                 dimensions: int,
                 bounds: Tuple[float, float],
                 population_size: int = 25,
                 beta0: float = 1.0,
                 gamma: float = 1.0,
                 alpha: float = 0.2,
                 alpha_decay: float = 0.95,
                 patience: int = 50,
                 tolerance: float = 1e-8,
                 is_minimization: bool = True):
        """
        Args:
            objective_function
            dimensions: problem dimensionality
            bounds: search space bounds (min, max)
            population_size: num of fireflies
            beta0: base attractiveness coefficient
            gamma: absorption coefficient (controls how fast attractiveness decreases)
            alpha: randomization parameter
            alpha_decay: decay rate for alpha (helps convergence)
            is_minimization: true for minimization, False for maximization
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.population_size = population_size

        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimization = is_minimization

        self.swarm = [Firefly(dimensions, bounds) for _ in range(population_size)]

        self.best_firefly = None
        self.best_fitness_history = []
        self.generation = 0
        self.patience_counter = 0

        self._evaluate_swarm()

    def _evaluate_swarm(self):
        for firefly in self.swarm:
            fitness = self.objective_function(firefly.position)
            firefly.calculate_intensity(fitness, self.is_minimization)

        self._update_best()

    def _update_best(self):
        if self.is_minimization:
            current_best = min(self.swarm, key=lambda f: f.fitness)
        else:
            current_best = max(self.swarm, key=lambda f: f.fitness)

        if self.best_firefly is None or self._is_better(current_best.fitness, self.best_firefly.fitness):
            if self.best_firefly is not None:
                improvement = abs(current_best.fitness - self.best_firefly.fitness)
                if improvement >= self.tolerance:
                    self._copy_firefly(current_best)
                    return True
                else:
                    return False
            else:
                self._copy_firefly(current_best)
                return True

        return False

    def _is_better(self, fitness1: float, fitness2: float) -> bool:
        if self.is_minimization:
            return fitness1 < fitness2
        else:
            return fitness1 > fitness2

    def _copy_firefly(self, source_firefly):
        if self.best_firefly is None:
            self.best_firefly = Firefly(self.dimensions, self.bounds)

        self.best_firefly.position = source_firefly.position.copy()
        self.best_firefly.fitness = source_firefly.fitness
        self.best_firefly.intensity = source_firefly.intensity

    def _move_fireflies(self):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i != j and self.swarm[j].intensity > self.swarm[i].intensity:
                    # random movement
                    random_factor = np.random.normal(0, 1, self.dimensions)

                    # move firefly i towards brighter firefly j
                    self.swarm[i].move_towards(
                        self.swarm[j], self.beta0, self.gamma,
                        self.alpha, random_factor
                    )

                    # Ensure firefly stays within bounds
                    self.swarm[i].clip_to_bounds(self.bounds)

                    # Re-evaluate fitness after movement
                    fitness = self.objective_function(self.swarm[i].position)
                    self.swarm[i].calculate_intensity(fitness, self.is_minimization)

    def _random_walk_brightest(self):
        """Apply random walk to fireflies that don't have brighter neighbors."""
        for firefly in self.swarm:
            # Check if this firefly is the brightest or among the brightest
            brighter_exists = any(other.intensity > firefly.intensity
                                  for other in self.swarm if other != firefly)

            if not brighter_exists:
                # Perform random walk
                random_factor = np.random.normal(0, 1, self.dimensions)
                firefly.random_walk(self.alpha, random_factor)
                firefly.clip_to_bounds(self.bounds)

                # Re-evaluate fitness
                fitness = self.objective_function(firefly.position)
                firefly.calculate_intensity(fitness, self.is_minimization)

    def optimize(self, max_generations: int, tolerance: float = 1e-6,
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the optimization process.

        Args:
            max_generations: Maximum number of generations
            tolerance: Convergence tolerance
            verbose: Whether to print progress

        Returns:
            Tuple of (best_position, best_fitness)
        """
        if verbose:
            print(f"Starting Firefly Algorithm Optimization")
            print(f"Population: {self.population_size}, Dimensions: {self.dimensions}")
            print(f"Parameters: β₀={self.beta0}, γ={self.gamma}, α={self.alpha}")
            print("-" * 60)

        for generation in range(max_generations):
            self.generation = generation

            self._move_fireflies()

            self._random_walk_brightest()

            improved = self._update_best()
            self.best_fitness_history.append(self.best_firefly.fitness)

            if verbose and (generation % (max_generations // 10) == 0 or generation == max_generations - 1):
                print(f"Generation {generation:4d}: Best fitness = {self.best_firefly.fitness:.8f}")

            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                break

            # Decay alpha for better convergence
            self.alpha *= self.alpha_decay

        if verbose:
            print("-" * 60)
            print(f"Optimization completed!")
            print(f"Best position: {self.best_firefly.position}")
            print(f"Best fitness: {self.best_firefly.fitness:.8f}")

        return self.best_firefly.position.copy(), self.best_firefly.fitness

    def plot_convergence(self):
        """Plot the convergence history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Firefly Algorithm Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_swarm_positions(self) -> np.ndarray:
        """Get current positions of all fireflies."""
        return np.array([firefly.position for firefly in self.swarm])

    def get_swarm_fitness(self) -> np.ndarray:
        """Get current fitness values of all fireflies."""
        return np.array([firefly.fitness for firefly in self.swarm])
