import numpy as np
import random
from typing import List, Callable
import matplotlib.pyplot as plt

from algorithms.ga.crossover import SinglePointCrossover, TwoPointCrossover, UniformCrossover
from algorithms.ga.individual import Individual
from algorithms.ga.mutation import GaussianMutation, FlipBitMutation
from algorithms.ga.population import PopulationInitializer
from algorithms.ga.selection import TournamentSelection, RouletteWheelSelection
from algorithms.ga.utils import SelectionMethod, CrossoverMethod, MutationMethod

import matplotlib
matplotlib.use('TkAgg') # remove for jupiter


class GeneticAlgorithm:

    def __init__(self,
                 population_size: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elitism_count: int = 2,
                 max_generations: int = 1000,
                 stagnation_limit: int = 50,
                 population_initializer: PopulationInitializer = None,
                 selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                 crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT,
                 mutation_method: MutationMethod = MutationMethod.GAUSSIAN,
                 tournament_size: int = 3,
                 mutation_sigma: float = 0.1):

        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit

        self.population_initializer = population_initializer
        self._setup_selection(selection_method, tournament_size)
        self._setup_crossover(crossover_method)
        self._setup_mutation(mutation_method, mutation_sigma)

        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.population = []
        self.best_individual = None

    def _setup_selection(self, method: SelectionMethod, tournament_size: int):
        if method == SelectionMethod.TOURNAMENT:
            self.selection_operator = TournamentSelection(tournament_size)
        elif method == SelectionMethod.ROULETTE_WHEEL:
            self.selection_operator = RouletteWheelSelection()
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _setup_crossover(self, method: CrossoverMethod):
        if method == CrossoverMethod.SINGLE_POINT:
            self.crossover_operator = SinglePointCrossover()
        elif method == CrossoverMethod.TWO_POINT:
            self.crossover_operator = TwoPointCrossover()
        elif method == CrossoverMethod.UNIFORM:
            self.crossover_operator = UniformCrossover()
        else:
            raise ValueError(f"Unknown crossover method: {method}")

    def _setup_mutation(self, method: MutationMethod, sigma: float):
        if method == MutationMethod.GAUSSIAN:
            self.mutation_operator = GaussianMutation(self.mutation_rate, sigma)
        elif method == MutationMethod.FLIP_BIT:
            self.mutation_operator = FlipBitMutation(self.mutation_rate)
        else:
            raise ValueError(f"Unknown mutation method: {method}")

    def evolve(self, fitness_function: Callable) -> Individual:
        self.population = self.population_initializer.initialize(self.population_size)

        for individual in self.population:
            individual.calculate_fitness(fitness_function)

        stagnation_counter = 0
        previous_best_fitness = float('-inf')

        for generation in range(self.max_generations):
            self.generation = generation

            self._update_statistics()

            current_best_fitness = max(ind.fitness for ind in self.population)
            if abs(current_best_fitness - previous_best_fitness) < 1e-10:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= self.stagnation_limit:
                print(f"Evolution stopped due to stagnation at generation {generation}")
                break

            previous_best_fitness = current_best_fitness

            # Create next generation
            self.population = self._create_next_generation(fitness_function)

            # Print progress
            if generation % 50 == 0:
                best_fitness = max(ind.fitness for ind in self.population)
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                print(f"Generation {generation}: Best={best_fitness:.6f}, Avg={avg_fitness:.6f}")

        # Return best individual found
        self.best_individual = max(self.population, key=lambda ind: ind.fitness)
        return self.best_individual

    def _create_next_generation(self, fitness_function: Callable) -> List[Individual]:
        """Create the next generation using GA operators"""
        # Sort population by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        next_generation = []

        # Elitism: Keep best individuals
        for i in range(self.elitism_count):
            next_generation.append(self.population[i].copy())

        # Generate offspring to fill the rest of the population
        while len(next_generation) < self.population_size:
            # Selection
            parents = self.selection_operator.select(self.population, 2)
            parent1, parent2 = parents[0], parents[1]

            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover_operator.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Mutation
            offspring1 = self.mutation_operator.mutate(offspring1)
            offspring2 = self.mutation_operator.mutate(offspring2)

            # Calculate fitness for new offspring
            offspring1.calculate_fitness(fitness_function)
            offspring2.calculate_fitness(fitness_function)

            # Add to next generation
            next_generation.extend([offspring1, offspring2])

        # Ensure exact population size
        return next_generation[:self.population_size]

    def _update_statistics(self):
        """Update fitness statistics for tracking"""
        fitnesses = [ind.fitness for ind in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(np.mean(fitnesses))

    def plot_evolution(self):
        """Plot the evolution progress"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, label='Best Fitness', color='red')
        plt.plot(self.avg_fitness_history, label='Average Fitness', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.best_fitness_history, color='red')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness Over Time')
        plt.grid(True)

        plt.tight_layout()
        plt.show()